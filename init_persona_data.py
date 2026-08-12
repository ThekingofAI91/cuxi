"""
初始化名人对话场景的数据

将 data/persona_chat/ 下的资料文件解析、分块、向量化后存入 ChromaDB。
用法：python init_persona_data.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import settings
from src.document_processing.parser import DocumentParser
from src.retrieval.chunker import AdaptiveChunker
from src.retrieval.embedder import get_embedder
from src.retrieval.source_profile import classify_source
from scenes.persona_chat.config import persona_chat_config


def load_character_data(character_id: str):
    """加载指定角色的数据到 ChromaDB"""

    character = persona_chat_config.characters.get(character_id)
    if not character:
        print(f"❌ 角色 '{character_id}' 不存在")
        return

    data_dir = Path(character.data_source)
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return

    collection_name = character.chroma_collection
    print(f"\n{'='*60}")
    print(f"🎭 正在加载角色: {character.name}")
    print(f"📂 数据目录: {data_dir}")
    print(f"📦 ChromaDB collection: {collection_name}")
    print(f"{'='*60}")

    # 收集所有文件：data_source 目录中的文件 + 父目录中的 md 文件
    files = [f for f in data_dir.iterdir() if f.is_file() and not f.name.startswith('.')]
    parent_dir = data_dir.parent
    if parent_dir != data_dir:
        md_files = [f for f in parent_dir.iterdir() if f.is_file() and f.suffix.lower() == '.md' and not f.name.startswith('.')]
        files.extend(md_files)
    if not files:
        print(f"⚠️ 数据目录为空: {data_dir}")
        return

    print(f"📄 找到 {len(files)} 个文件")

    parser = DocumentParser()
    chunker = AdaptiveChunker(
        min_size=settings.chunk_size // 2,
        max_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    all_chunks = []

    for file_path in files:
        print(f"\n  处理: {file_path.name}")
        try:
            elements = parser.parse(file_path)
            print(f"    解析: {len(elements)} 个元素")

            chunks = chunker.chunk(elements, source=file_path.name)
            # 语料来源类型打标（original/oral/secondary/artificial/anchor）：
            # 检索阶段按类型加权（口述体优先、二手解读降权），提前写入元数据。
            for c in chunks:
                c.metadata.setdefault("source_type", classify_source(c.metadata.get("source", "")))
            print(f"    分块: {len(chunks)} 个块")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"    ❌ 处理失败: {e}")

    if not all_chunks:
        print("\n⚠️ 没有生成任何文档块")
        return

    print(f"\n📊 总计: {len(all_chunks)} 个文档块")

    # 向量化
    print("🔢 正在向量化...")
    embedder = get_embedder()
    vectors, metadatas = embedder.embed_documents_with_metadata(all_chunks)

    # 存入 ChromaDB
    print(f"💾 正在存入 ChromaDB ({collection_name})...")

    # 清除旧数据
    print(f"🗑️ 正在清除旧数据...")
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection(name=collection_name)
        print(f"   已清除旧 collection: {collection_name}")
    except Exception:
        print(f"   无旧 collection，继续...")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        batch_ids = [f"{character_id}_{j}" for j in range(i, batch_end)]
        batch_texts = [all_chunks[j].page_content for j in range(i, batch_end)]
        batch_metadatas = [all_chunks[j].metadata for j in range(i, batch_end)]
        batch_vectors = vectors[i:batch_end]

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_vectors,
            metadatas=batch_metadatas,
        )

    print(f"\n✅ 角色 '{character.name}' 的数据加载完成!")
    print(f"   Collection: {collection_name}")
    print(f"   文档块数: {len(all_chunks)}")
    print(f"   向量维度: {len(vectors[0]) if vectors else 'N/A'}")


def main():
    """初始化所有角色的数据"""
    print("🎭 名人对话场景 — 数据初始化")
    print(f"   可用角色: {list(persona_chat_config.characters.keys())}")

    for character_id in persona_chat_config.characters:
        load_character_data(character_id)

    print(f"\n{'='*60}")
    print("🎉 所有角色数据初始化完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
