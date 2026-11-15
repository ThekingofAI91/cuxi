"""嵌入不可用时只用真实 BM25 资料，入库则停止写入。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from src.retrieval import advanced_search, ingest
from src.retrieval.embedder import EmbeddingError
from src.retrieval.trace import RetrievalTrace


@pytest.fixture()
def retrieval(monkeypatch):
    docs = [Document(page_content="原型是集体无意识的内容。", metadata={"source": "book.md"})]
    embedder = SimpleNamespace(embed_documents=Mock(side_effect=EmbeddingError("embedding unavailable")))
    index = SimpleNamespace(document_count=1, search=Mock(return_value=[(docs[0], 1.0)]))
    collection = SimpleNamespace(name="test-library", query=Mock(side_effect=AssertionError("invalid vector query")))
    monkeypatch.setattr("src.retrieval.embedder.get_embedder", lambda: embedder)
    monkeypatch.setattr(advanced_search, "_ensure_bm25_ready", lambda _: (docs, index))
    return docs, embedder, index, collection


def test_embedding_failure_keeps_bm25_and_records_degradation(retrieval, caplog):
    docs, _, _, collection = retrieval
    trace = RetrievalTrace()
    results, contexts = asyncio.run(advanced_search.advanced_retrieval(
        "原型是什么", collection, top_k=1, use_rerank=False, trace_out=trace,
    ))

    assert contexts == [docs[0].page_content]
    assert results[0].metadata["retrieval_mode"] == "bm25_only"
    assert [board["class"] for board in trace.boards] == ["bm25"]
    assert trace.to_dict()["degradation"] == {"strategy": "bm25_only", "reason": "embedding_unavailable"}
    assert "fallback=bm25_only" in caplog.text
    collection.query.assert_not_called()


def test_embedding_failure_without_bm25_is_reported(retrieval):
    _, _, _, collection = retrieval
    with pytest.raises(EmbeddingError):
        asyncio.run(advanced_search.advanced_retrieval("问题", collection, use_bm25=False))


def test_both_retrieval_routes_failing_is_reported(retrieval):
    _, _, index, collection = retrieval
    index.search.side_effect = RuntimeError("BM25 failed")
    with pytest.raises(EmbeddingError, match="BM25 检索也失败"):
        asyncio.run(advanced_search.advanced_retrieval("问题", collection))


def test_other_vector_backend_errors_are_not_mislabeled_as_embedding_failures(retrieval):
    _, embedder, _, collection = retrieval
    embedder.embed_documents.side_effect = None
    embedder.embed_documents.return_value = [[1.0, 0.0, 0.0]]
    collection.query.side_effect = RuntimeError("vector database unavailable")
    with pytest.raises(RuntimeError, match="vector database unavailable"):
        asyncio.run(advanced_search.advanced_retrieval("问题", collection))


def test_ingestion_failure_never_deletes_or_writes_a_collection(monkeypatch):
    doc = Document(page_content="待入库正文", metadata={})
    embedder = SimpleNamespace(embed_documents_with_metadata=Mock(side_effect=EmbeddingError("inference failed")))
    client = Mock()
    monkeypatch.setattr(ingest, "chunk_text", lambda *args, **kwargs: [doc])
    monkeypatch.setattr(ingest, "get_embedder", lambda: embedder)
    monkeypatch.setattr("framework.runtime.get_chroma_client", lambda: client)

    with pytest.raises(EmbeddingError):
        ingest.ingest_texts("existing-library", "test", "待入库正文")

    assert client.mock_calls == []


def test_corpus_rebuild_preflights_first_batch_before_clearing_old_data(tmp_path, monkeypatch):
    import init_persona_data as init

    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    source = source_dir / "book.md"
    source.write_text("待入库正文", encoding="utf-8")
    character = SimpleNamespace(name="test", data_source=str(source_dir), chroma_collection="existing-library")
    docs = [Document(page_content="待入库正文", metadata={"source_type": "original"})]
    embedder = SimpleNamespace(embed_documents=Mock(side_effect=EmbeddingError("inference failed")))
    client = Mock()
    monkeypatch.setitem(init.persona_chat_config.characters, "embedding-test", character)
    monkeypatch.setattr(init, "collect_corpus_files", lambda _: ([source], {}, [], []))
    monkeypatch.setattr(init, "build_chunks", lambda *args, **kwargs: (
        docs, {"ok": 1, "failed": 0, "secs": 0.0}, [],
    ))
    monkeypatch.setattr(init, "clean_chunks", lambda chunks: (chunks, {}))
    monkeypatch.setattr(init, "get_embedder", lambda: embedder)
    monkeypatch.setattr(init, "_client", lambda: client)

    with pytest.raises(EmbeddingError):
        init.load_character_data("embedding-test")

    assert client.mock_calls == []
