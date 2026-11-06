"""LLM-as-Judge 的五档评分与结果文件约定。"""

import os
import re
from pathlib import Path
from typing import Optional


SCORE_MIN = 1
SCORE_MAX = 5
SCORE_SCALE = {"min": SCORE_MIN, "max": SCORE_MAX}
METRICS = (
    "faithfulness",
    "relevancy",
    "context_precision",
    "key_points_coverage",
    "voice_consistency",
)


def get_eval_result_path() -> Path:
    """新评分另存文件；主评测和辅助脚本共用 EVAL_RESULT_FILE。"""
    root = Path(__file__).resolve().parents[2]
    return root / "tests" / os.environ.get("EVAL_RESULT_FILE", "eval_results_5point.json")


def get_score_scale(data: dict) -> dict:
    """未标注分制的历史结果采用原来的 0–10 分制。"""
    return data.get("score_scale", {"min": 0, "max": 10})


def require_current_score_scale(data: dict, path: Path) -> None:
    """续跑或补评前检查口径，防止把旧分数混入五分制结果。"""
    if data.get("results") and get_score_scale(data) != SCORE_SCALE:
        raise ValueError(
            f"{path} 中的评分口径与当前 1–5 分制不一致；"
            "请用 EVAL_RESULT_FILE 指定新的结果文件，保留历史结果后重新评测。"
        )


def normalize_score(value) -> Optional[int]:
    """接受 1–5 整数及 '4分'、'4/5' 等写法；无效值不计入均分。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*([+-]?\d+(?:\.\d+)?)\s*(?:分)?"
            r"(?:\s*/\s*(\d+(?:\.\d+)?)\s*(?:分)?)?\s*",
            value,
        )
        if not match:
            return None
        if match.group(2) is not None and float(match.group(2)) != SCORE_MAX:
            return None
        score = float(match.group(1))
    elif isinstance(value, (int, float)):
        score = value
    else:
        return None
    if not SCORE_MIN <= score <= SCORE_MAX:
        return None
    return int(score) if score == int(score) else None
