"""
memory.py — 用户长期记忆模块

让系统跨会话记住"这个人"：背景（备考/职业/所在阶段）、偏好（喜欢简洁还是深入）、
持续关注的主题。对话时检索相关记忆注入提示词，体验从「金鱼记忆」变「老朋友」。

设计与隐私边界：
- 仅对登录用户启用（user_key = "u:<账号id>"）；匿名用户无跨会话身份，不建记忆。
- 只存提炼后的事实条目（≤40 字/条），绝不存对话原文。
- 账号注销时全量删除（routes.auth_delete_account 调 delete_user）。
- 提取在回答返回后异步执行（与摘要压缩同一模式），不占用户等待时间。
- 去重靠 embedding 近似（同 kind 余弦 ≥ 0.85 视为同一条 → 更新而非新增）。

存储：同步 sqlite3 + WAL + 全局锁（与 session_store 同风格，demo 规模足够）。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.core.config import settings

_EXTRACT_PROMPT = """你在为一名人对话应用维护「用户长期记忆」。根据这轮对话，判断是否出现了值得长期记住的**用户侧信息**。

规则：
- 只记用户本人的信息：背景（备考/职业/所处阶段）、目标、偏好（喜欢简洁还是深入、怎么称呼）、持续关注的主题
- 忽略：寒暄、知识概念本身（那是知识库的事）、回答的内容、一次性上下文（"刚才那个词"）
- 每条 ≤ 40 字，一句一个事实；不确定的不记，不要过度推断
- kind 取值：profile = 稳定背景/偏好（长期适用）；topic = 用户正在关注的主题
- 已有记忆里已经有的不要重复输出；如果情况变化了（如"考研"改成"二战"），输出新表述
- 没有值得记的就返回空列表

已有记忆：
{existing}

本轮对话：
用户：{query}
回答：{answer}

输出严格 JSON（不要输出 JSON 以外的任何文字）：
{{"memories": [{{"kind": "profile", "text": "……"}}]}}"""

_PROFILE_CAP = 8     # profile 条数上限（超出淘汰最旧的）
_TOPIC_CAP = 40      # topic 条数上限
_TOPICAL_TOP_K = 3   # 每次注入的主题记忆条数上限
_TOPICAL_MIN_SIM = 0.55
_REPLACE_SIM = 0.85  # 同 kind 近似视为同一条 → 更新

# 提取任务引用集：防止后台任务被 GC（与 advanced_search._bg_rewrite_tasks 同模式）
_bg_extract_tasks: set = set()


def _embed_texts(texts: list[str]) -> np.ndarray:
    """向量化（独立小函数便于测试替换）"""
    from src.retrieval.embedder import get_embedder
    vecs = get_embedder().embed_documents(texts)
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _get_extract_llm():
    """记忆提取用 LLM（独立小函数便于测试替换，避免测试打到真实 API）"""
    from src.core.llm import get_chat_llm
    return get_chat_llm(temperature=0.1, max_tokens=400)


class MemoryStore:
    """用户长期记忆存储（同步 sqlite3 + WAL + 全局锁）"""

    def __init__(self, db_path: str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_key TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_key)")
            self._conn.commit()

    def all_for_user(self, user_key: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, text, vec, updated_at FROM memories WHERE user_key = ?",
                (user_key,),
            ).fetchall()
        return [
            {
                "id": r["id"], "kind": r["kind"], "text": r["text"],
                "vec": np.frombuffer(r["vec"], dtype=np.float32),
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def insert(self, user_key: str, kind: str, text: str, vec: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (user_key, kind, text, vec, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_key, kind, text, vec.astype(np.float32).tobytes(), now, now),
            )
            self._conn.commit()

    def update_text(self, mem_id: int, text: str, vec: np.ndarray) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET text = ?, vec = ?, updated_at = ? WHERE id = ?",
                (text, vec.astype(np.float32).tobytes(), time.time(), mem_id),
            )
            self._conn.commit()

    def delete(self, mem_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            self._conn.commit()

    def trim(self, user_key: str, kind: str, cap: int) -> None:
        """按 kind 限条数，超出淘汰最旧的"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE user_key = ? AND kind = ? "
                "ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?",
                (user_key, kind, cap),
            ).fetchall()
            for r in rows:
                self._conn.execute("DELETE FROM memories WHERE id = ?", (r["id"],))
            self._conn.commit()

    def delete_user(self, user_key: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE user_key = ?", (user_key,))
            self._conn.commit()
            return cur.rowcount


_store: Optional[MemoryStore] = None
_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = MemoryStore(settings.memory_db_path)
    return _store


# ============================================================
# 提取（回答返回后异步执行）
# ============================================================

def _extract_sync_parse(content: str) -> list[dict]:
    """解析判官…不对，解析提取结果；失败返回空列表"""
    from src.core.llm_json import extract_json
    try:
        obj = extract_json(content)
        items = obj.get("memories") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            return []
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            text = str(it.get("text", "")).strip()
            kind = str(it.get("kind", "topic")).strip()
            if text and kind in ("profile", "topic") and len(text) <= 60:
                out.append({"kind": kind, "text": text})
        return out
    except Exception:
        return []


async def extract_and_store(user_key: str, query: str, answer: str) -> None:
    """从一轮对话中提取用户侧事实并合并进记忆库（不抛异常，后台任务）"""
    if not getattr(settings, "memory_enabled", True) or not user_key:
        return
    try:
        store = get_memory_store()
        existing = store.all_for_user(user_key)
        existing_text = "\n".join(
            f"{i}. [{m['kind']}] {m['text']}" for i, m in enumerate(existing, 1)
        ) or "（暂无）"

        from src.core.llm import ainvoke_nonempty
        resp = await ainvoke_nonempty(_get_extract_llm(), [("user", _EXTRACT_PROMPT.format(
            existing=existing_text,
            query=query[:600],
            answer=(answer or "")[:1200],
        ))])
        items = _extract_sync_parse(getattr(resp, "content", "") or "")
        if not items:
            return

        # 合并：与已有同 kind 条目近似（余弦 ≥ 0.85）→ 更新该条；否则新增
        merge_items(existing, items, store, user_key)
    except Exception as e:
        print(f"[Memory] 记忆提取失败（忽略）: {e}")


def merge_items(existing: list[dict], items: list[dict], store: MemoryStore, user_key: str) -> None:
    """新条目合并进库：近似去重（更新）或新增；随后按 kind 限条数"""
    new_texts = [it["text"] for it in items]
    new_vecs = _embed_texts(new_texts)
    for it, vec in zip(items, new_vecs):
        pool = [m for m in existing if m["kind"] == it["kind"]]
        best, best_sim = None, 0.0
        for m in pool:
            sim = float(np.dot(vec, m["vec"]))
            if sim > best_sim:
                best, best_sim = m, sim
        if best is not None and best_sim >= _REPLACE_SIM:
            if best["text"] != it["text"]:
                store.update_text(best["id"], it["text"], vec)
                best["text"], best["vec"], best["updated_at"] = it["text"], vec, time.time()
        else:
            store.insert(user_key, it["kind"], it["text"], vec)
            existing.append({"id": None, "kind": it["kind"], "text": it["text"],
                             "vec": vec, "updated_at": time.time()})
    store.trim(user_key, "profile", _PROFILE_CAP)
    store.trim(user_key, "topic", _TOPIC_CAP)


# ============================================================
# 检索与注入
# ============================================================

async def retrieve_memory_block(user_key: str, query: str) -> str:
    """取回与当前问题相关的记忆并渲染成提示词块；无记忆返回空串"""
    if not getattr(settings, "memory_enabled", True) or not user_key:
        return ""
    try:
        store = get_memory_store()
        all_items = await asyncio.to_thread(store.all_for_user, user_key)
        if not all_items:
            return ""
        qvec = _embed_texts([query])[0]

        profiles = sorted(
            (m for m in all_items if m["kind"] == "profile"),
            key=lambda m: m["updated_at"], reverse=True,
        )[:5]
        topics = [m for m in all_items if m["kind"] == "topic"]
        scored = sorted(
            ((float(np.dot(qvec, m["vec"])), m) for m in topics),
            key=lambda t: t[0], reverse=True,
        )
        topical = [m for sim, m in scored[:_TOPICAL_TOP_K] if sim >= _TOPICAL_MIN_SIM]

        seen, lines = set(), []
        for m in (profiles + topical):
            if m["text"] in seen:
                continue
            seen.add(m["text"])
            lines.append(f"- {m['text']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[Memory] 记忆检索失败（忽略）: {e}")
        return ""


def format_memory_directive(block: str) -> str:
    """渲染为系统提示词块"""
    return (
        "【关于这位用户的长期记忆】回答可以自然贴合这些背景（如对方所处的阶段、偏好），"
        "但绝不要罗列条目、也不要说「我记得你说过」——把记忆当成你自己对这位朋友的了解，"
        "自然带出即可。与当前问题无关就忽略。\n" + block
    )
