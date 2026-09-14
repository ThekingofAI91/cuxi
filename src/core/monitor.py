"""
成本 / 错误监控：按请求记录 token 估算与费用，SQLite 落盘 + 汇总查询。

估算口径（DeepSeek 中文约 0.75 token/字）：
    prompt_tokens ≈ prompt_chars * 0.75
    completion_tokens ≈ answer_chars * 0.75
费用按配置单价：cost_input_per_1m / cost_output_per_1m（元 / 百万 token）。

有 usage_metadata 的调用方（LLM 返回）可以直接传入真实 token 数，
未提供时自动回退到字数估算。
"""

import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class UsageMonitor:
    def __init__(
        self,
        db_path: str,
        cost_input_per_1m: float = 1.0,
        cost_output_per_1m: float = 2.0,
    ):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cost_in = cost_input_per_1m
        self._cost_out = cost_output_per_1m
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    request_id TEXT,
                    character TEXT,
                    prompt_chars INTEGER DEFAULT 0,
                    answer_chars INTEGER DEFAULT 0,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0,
                    latency_ms INTEGER DEFAULT 0,
                    cache_hit INTEGER DEFAULT 0,
                    error INTEGER DEFAULT 0,
                    stages TEXT DEFAULT ''
                )
                """
            )
            # 旧库迁移：补 stages 列（阶段耗时瀑布，JSON 字符串）
            cols = [r[1] for r in self._conn.execute("PRAGMA table_info(usage_log)").fetchall()]
            if "stages" not in cols:
                self._conn.execute("ALTER TABLE usage_log ADD COLUMN stages TEXT DEFAULT ''")
            self._conn.commit()

    @staticmethod
    def _estimate_tokens(chars: int) -> int:
        return int(chars * 0.75)

    def record(
        self,
        request_id: str,
        character: str,
        prompt_chars: int = 0,
        answer_chars: int = 0,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: int = 0,
        cache_hit: bool = False,
        error: bool = False,
        stages: Optional[dict] = None,
    ) -> None:
        pt = prompt_tokens if prompt_tokens is not None else self._estimate_tokens(prompt_chars)
        ct = completion_tokens if completion_tokens is not None else self._estimate_tokens(answer_chars)
        cost = pt / 1_000_000 * self._cost_in + ct / 1_000_000 * self._cost_out
        try:
            import json as _json

            stages_json = _json.dumps(stages, ensure_ascii=False) if stages else ""
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO usage_log
                    (ts, request_id, character, prompt_chars, answer_chars,
                     prompt_tokens, completion_tokens, cost, latency_ms, cache_hit, error, stages)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        time.time(),
                        request_id,
                        character or "unknown",
                        prompt_chars,
                        answer_chars,
                        pt,
                        ct,
                        round(cost, 6),
                        latency_ms,
                        1 if cache_hit else 0,
                        1 if error else 0,
                        stages_json,
                    ),
                )
                self._conn.commit()
        except Exception as e:
            print(f"[Monitor] ⚠️ 记录失败: {e}")

    def summary(self, since_ts: float) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM usage_log WHERE ts >= ?", (since_ts,)
            ).fetchall()
        cols = [d[0] for d in self._conn.execute("SELECT * FROM usage_log LIMIT 0").description]
        items = [dict(zip(cols, r)) for r in rows]
        total_cost = sum(i["cost"] for i in items)
        total_tokens = sum(i["prompt_tokens"] + i["completion_tokens"] for i in items)
        total_requests = len(items)
        errors = sum(1 for i in items if i["error"])
        cache_hits = sum(1 for i in items if i["cache_hit"])
        latencies = [i["latency_ms"] for i in items if i["latency_ms"]]
        avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0

        per_character: dict[str, dict] = {}
        for i in items:
            ch = i["character"] or "unknown"
            d = per_character.setdefault(ch, {"requests": 0, "cost": 0.0, "errors": 0})
            d["requests"] += 1
            d["cost"] = round(d["cost"] + i["cost"], 4)
            d["errors"] += i["error"]

        # 最近带阶段耗时的请求（延迟瀑布用，最新的在前）
        recent_stages: list[dict] = []
        for i in reversed(items):
            if not i.get("stages"):
                continue
            try:
                import json as _json

                recent_stages.append({
                    "ts": i["ts"],
                    "character": i["character"],
                    "latency_ms": i["latency_ms"],
                    "stages": _json.loads(i["stages"]),
                })
            except Exception:
                continue
            if len(recent_stages) >= 30:
                break

        return {
            "requests": total_requests,
            "errors": errors,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / total_requests, 3) if total_requests else 0,
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_latency_ms": avg_latency,
            "per_character": per_character,
            "recent_stages": recent_stages,
        }

    def since_start_of_day(self) -> float:
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()


_monitor: Optional[UsageMonitor] = None


def get_monitor() -> UsageMonitor:
    global _monitor
    if _monitor is None:
        from src.core.config import settings

        _monitor = UsageMonitor(
            settings.monitor_db_path,
            cost_input_per_1m=settings.cost_input_per_1m,
            cost_output_per_1m=settings.cost_output_per_1m,
        )
    return _monitor
