"""嵌入失败不得返回或缓存假向量；所有模型均为离线替身。"""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.documents import Document

from src.retrieval.embedder import Embedder, EmbeddingError


def _ready_embedder(tmp_path, output=None):
    embedder = Embedder(model_name="test-model", cache_dir=str(tmp_path))
    embedder._dimension = 3
    embedder._model = SimpleNamespace(encode=Mock(
        return_value=output if output is not None else np.array([[1.0, 0.0, 0.0]]),
    ))
    return embedder


def _patch_loader(monkeypatch, loader):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=loader))


def test_loading_failure_raises_without_vectors_or_cache(tmp_path, monkeypatch):
    failure = RuntimeError("model unavailable")
    _patch_loader(monkeypatch, Mock(side_effect=failure))
    embedder = Embedder(model_name="missing-model", cache_dir=str(tmp_path))

    with pytest.raises(EmbeddingError, match="加载失败") as caught:
        embedder.embed_query("测试问题")

    assert caught.value.__cause__ is failure
    assert embedder._model is None and embedder._dimension is None
    assert list(tmp_path.iterdir()) == []


def test_failed_initialization_can_retry_without_a_partial_model(tmp_path, monkeypatch):
    broken = SimpleNamespace(get_embedding_dimension=Mock(side_effect=ValueError("bad dimension")))
    healthy = SimpleNamespace(
        get_embedding_dimension=lambda: 3,
        encode=lambda texts, **kwargs: np.tile([1.0, 0.0, 0.0], (len(texts), 1)),
    )
    loader = Mock(side_effect=[broken, healthy])
    _patch_loader(monkeypatch, loader)
    embedder = Embedder(model_name="retry-model", cache_dir=str(tmp_path))

    with pytest.raises(EmbeddingError):
        embedder.embed_query("测试问题")
    assert embedder._model is None
    assert embedder.embed_query("测试问题") == [1.0, 0.0, 0.0]
    assert loader.call_count == 2


def test_encoding_failure_raises_instead_of_returning_seeded_vectors(tmp_path):
    embedder = _ready_embedder(tmp_path)
    embedder._model.encode.side_effect = RuntimeError("inference failed")

    for query in ("第一个问题", "完全不同的问题"):
        with pytest.raises(EmbeddingError, match="向量化失败"):
            embedder.embed_query(query)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("output", [
    np.array([[1.0, 0.0]]),
    np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    np.array([[np.nan, 0.0, 0.0]]),
    np.array([[np.inf, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0]]),
])
def test_invalid_model_output_is_not_cached(tmp_path, output):
    embedder = _ready_embedder(tmp_path, output)
    with pytest.raises(EmbeddingError):
        embedder.embed_query("测试问题")
    assert list(tmp_path.iterdir()) == []


def test_successful_vectors_and_metadata_are_preserved_and_cached(tmp_path):
    embedder = _ready_embedder(tmp_path)
    doc = Document(page_content="原文", metadata={"source": "book.md"})

    vectors, metadata = embedder.embed_documents_with_metadata([doc])
    assert vectors == [[1.0, 0.0, 0.0]] and metadata == [doc.metadata]
    assert embedder.embed_query("原文") == vectors[0]
    embedder._model.encode.assert_called_once()
    assert embedder.embed_documents([]) == []


def test_legacy_cache_is_ignored_and_models_have_separate_keys(tmp_path):
    text = "缓存问题"
    legacy_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    (tmp_path / f"{legacy_key}.json").write_text(json.dumps([0.2, 0.3, 0.4]), encoding="utf-8")
    embedder = _ready_embedder(tmp_path)

    assert embedder.embed_query(text) == [1.0, 0.0, 0.0]
    embedder._model.encode.assert_called_once()
    other = Embedder(model_name="another-model", cache_dir=str(tmp_path))
    assert other._check_cache(text) is None
