"""
auto_pipeline.py — 中转恢复守望者 + 评估管线自动执行

后台运行：每 PROBE_INTERVAL 秒单发探测一次（不重试，避免自我触发限流），
连续 2 次有内容判定恢复，随后按序执行：
  1. 确认服务在跑（不通则启动 main.py）
  2. 评估集扩充（generate_eval_set.py --delay 8）
  3. 全量基线评估（EVAL_JUDGE_GAP_SEC=25, --pace-sec 20）
  4. 重启建图战役（build_graphs_all.py --delay 8，断点续传）
  5. 项目日记追加优化七十五条目
全部输出重定向到 output/auto_pipeline.log（本脚本自身）与各子任务日志。

用法：.venv/Scripts/python.exe -u scripts/auto_pipeline.py &
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

PROBE_INTERVAL = 180      # 3 分钟
MAX_WAIT_ROUNDS = 60      # 最多守望 3 小时
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


async def probe_once() -> bool:
    """单发探测：有内容=True。绝不内部重试（避免连发自撞限流）。"""
    from src.core.llm import get_chat_llm

    llm = get_chat_llm(temperature=0, max_tokens=8, max_retries=0)
    try:
        r = await llm.ainvoke([("user", "回复：好")])
        return bool(str(getattr(r, "content", "")).strip())
    except Exception:
        return False


async def wait_for_recovery() -> bool:
    consecutive = 0
    for i in range(1, MAX_WAIT_ROUNDS + 1):
        ok = await probe_once()
        consecutive = consecutive + 1 if ok else 0
        log(f"探测 #{i}: {'有内容' if ok else '空/异常'}（连续 {consecutive}/2）")
        if consecutive >= 2:
            log("判定中转恢复，开始执行管线")
            return True
        await asyncio.sleep(PROBE_INTERVAL)
    log("守望超时（3 小时），放弃")
    return False


def run_step(name: str, cmd: list[str], env_extra: dict | None = None,
             timeout_s: int = 3600) -> bool:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    log(f">>> {name}: {' '.join(cmd)}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout_s)
        tail = (r.stdout or "")[-800:]
        log(f"<<< {name}: exit={r.returncode} 耗时 {(time.time()-t0)/60:.1f} 分钟\n--- 输出尾部 ---\n{tail}")
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"<<< {name}: 超时（{timeout_s}s）")
        return False
    except Exception as e:
        log(f"<<< {name}: 异常 {e}")
        return False


def main() -> None:
    log("守望者启动")
    if not asyncio.run(wait_for_recovery()):
        Path("output").mkdir(exist_ok=True)
        (ROOT / "output" / "auto_pipeline_done.txt").write_text(
            f"守望超时未恢复 {datetime.now()}", encoding="utf-8")
        return

    # 1) 服务
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        log("服务已在运行")
    except Exception:
        log("服务未运行，启动 main.py")
        subprocess.Popen([PY, "-u", "main.py"],
                         stdout=open("output/server_auto.log", "ab"),
                         stderr=subprocess.STDOUT, cwd=str(ROOT))
        time.sleep(60)

    # 2) 评估集扩充
    ok_gen = run_step("评估集扩充", [PY, "-u", "scripts/generate_eval_set.py", "--delay", "8"],
                      timeout_s=2400)
    n_questions = 0
    p = ROOT / "tests" / "eval_dataset_expanded.json"
    if p.exists():
        try:
            n_questions = len(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    log(f"评估集扩充完成：{n_questions} 题（ok={ok_gen}）")

    # 3) 全量基线评估
    ok_eval = run_step("全量基线评估",
                       [PY, "-u", "tests/eval_rag.py", "--pace-sec", "20"],
                       env_extra={"EVAL_JUDGE_GAP_SEC": "25"}, timeout_s=5400)
    summary = {}
    try:
        summary = json.loads((ROOT / "tests" / "eval_results.json").read_text(encoding="utf-8"))["summary"]
    except Exception:
        pass
    log(f"基线评估 summary: {json.dumps(summary, ensure_ascii=False)}")

    # 4) 建图战役重启（后台，断点续传）
    graphs = ROOT / "data" / "persona_chat" / "graphs"
    have = [g for g in ("persona_jung.json", "persona_adler.json", "persona_wangyangming.json")
            if (graphs / g).exists()] if graphs.exists() else []
    if len(have) < 3:
        log(f"重启建图战役（已有最终图谱: {have or '无'}）")
        subprocess.Popen([PY, "-u", "scripts/build_graphs_all.py", "--delay", "8"],
                         stdout=open("output/graph_campaign3.log", "ab"),
                         stderr=subprocess.STDOUT, cwd=str(ROOT))

    # 5) 日记
    entry = (
        f"\n\n---\n\n#### 优化七十五（自动任务）：中转恢复，评估管线自动执行\n"
        f"- **触发**：守望者探测到中转恢复（连续 2 次有内容），自动按序执行。\n"
        f"- **评估集扩充**：{'已完成' if ok_gen else '未完成'}，{n_questions} 题（tests/eval_dataset_expanded.json，覆盖 jung/adler/wangyangming）。\n"
        f"- **全量基线（含扩充题 + 合并判官）**：{'已完成' if ok_eval else '未完成'}，"
        f"忠实度 {summary.get('avg_faithfulness')} | 相关性 {summary.get('avg_relevancy')} | "
        f"上下文 {summary.get('avg_context_precision')} | 要点 {summary.get('avg_key_points_coverage')} | "
        f"有效评判数 {summary.get('judged_counts')} | 题数 {summary.get('total_questions')}。\n"
        f"- **建图战役**：已后台重启续传（--delay 8），进度见 output/graph_campaign3.log。\n"
        f"- **待办**：四项检索实验（改写开关 / RERANK_CANDIDATES=10 / 上下文压缩）待基线稳定后逐项 A/B。\n"
        f"- {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    )
    with open(ROOT / "项目日记.md", "a", encoding="utf-8") as f:
        f.write(entry)
    (ROOT / "output" / "auto_pipeline_done.txt").write_text(
        f"管线完成 {datetime.now()}", encoding="utf-8")
    log("管线全部完成")


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    main()
