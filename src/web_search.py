"""
web_search.py — 轻量网络搜索（零额外依赖，仅用标准库 urllib）

用途：创建"热门人物"自建角色时，抓取公开资料（生平、代表观点、名言），
交给 LLM 辅助生成人设 prompt 与背景知识库。

设计要点：
- 不引入 requests / bs4 等新依赖，直接用 urllib.request（标准库）。
- 多引擎兜底：auto 模式依次尝试 bing -> baidu -> ddg，任一成功即采用。
- 全程容错：网络异常 / 被墙 / 解析失败都返回 []，绝不抛出，
  调用方据此降级为"仅用用户提交的背景"。
- 仅做只读 GET；搜索结果只取文本片段，不上传用户任何隐私。
"""

from __future__ import annotations

import re
import ssl
import urllib.parse
import urllib.request
from typing import Optional

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# 各引擎搜索 URL 模板（q 为已编码查询）
_ENGINE_URLS = {
    "bing": "https://www.bing.com/search?q={q}&setlang=zh-CN&cc=CN",
    "baidu": "https://www.baidu.com/s?wd={q}",
    "ddg": "https://lite.duckduckgo.com/lite/?q={q}",
}

# auto 模式尝试顺序（国内环境 bing/baidu 通常比 ddg 可达）
_AUTO_ORDER = ["bing", "baidu", "ddg"]


def _make_context() -> ssl.SSLContext:
    """跳过证书校验的上下文（只读搜索，降低国内部分站点证书问题导致的失败）"""
    try:
        ctx = ssl.create_unverified_context()
    except Exception:
        ctx = ssl.create_default_context()
    return ctx


def _fetch_html(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml",
    })
    with urllib.request.urlopen(req, timeout=timeout, context=_make_context()) as resp:
        raw = resp.read()
    # 尽量按声明的编码解码；失败再用 ignore 兜底
    charset = resp.headers.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="ignore")
    except Exception:
        return raw.decode("utf-8", errors="ignore")


def _strip_tags(html: str) -> str:
    """去掉 script/style 与所有标签，压缩空白，得到纯文本"""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"&[a-z]+;", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def _extract_blocks(html: str) -> list[str]:
    """从搜索结果页提取候选文本片段（按引擎常见结果容器；失败则回退到 <p>）"""
    chunks: list[str] = []
    # Bing：<li class="b_algo"> 整块
    for m in re.finditer(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?</li>',
                         html, re.DOTALL | re.IGNORECASE):
        chunks.append(_strip_tags(m.group(0)))
    # Baidu：<div class="c-abstract ..."> 整块
    for m in re.finditer(r'<div[^>]*class="[^"]*c-abstract[^"]*"[^>]*>.*?</div>',
                         html, re.DOTALL | re.IGNORECASE):
        chunks.append(_strip_tags(m.group(0)))
    # DDG lite：<td class="result-snippet">
    for m in re.finditer(r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>.*?</td>',
                         html, re.DOTALL | re.IGNORECASE):
        chunks.append(_strip_tags(m.group(0)))
    # 通用兜底：所有 <p> 段落
    for m in re.finditer(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE):
        chunks.append(_strip_tags(m.group(1)))
    return chunks


def _clean_snippets(chunks: list[str], max_results: int) -> list[str]:
    """按长度过滤、去重、截断，得到干净的搜索片段"""
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        c = c.strip()
        if not c or len(c) < 25 or len(c) > 360:
            continue
        key = c[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= max_results:
            break
    return out


def _search_once(engine: str, query: str, max_results: int, timeout: float) -> list[str]:
    url = _ENGINE_URLS[engine].format(q=urllib.parse.quote_plus(query))
    html = _fetch_html(url, timeout)
    blocks = _extract_blocks(html)
    return _clean_snippets(blocks, max_results)


def web_research(
    query: str,
    max_results: int = 8,
    timeout: float = 8.0,
    provider: str = "auto",
) -> list[str]:
    """
    搜索 query 并返回文本片段列表（已去重、截断）。

    任何异常都返回空列表（调用方据此降级）。
    """
    if not query or not query.strip():
        return []
    engines = _AUTO_ORDER if provider == "auto" else [provider]
    last_err: Optional[str] = None
    for engine in engines:
        if engine not in _ENGINE_URLS:
            continue
        try:
            snippets = _search_once(engine, query, max_results, timeout)
            if snippets:
                return snippets
        except Exception as e:  # 单引擎失败不致命，继续下一个
            last_err = f"{engine}: {e}"
            continue
    if last_err:
        print(f"[web_search] 所有引擎失败（末次: {last_err}）")
    return []


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "卡尔 荣格"
    res = web_research(q, max_results=5)
    for i, s in enumerate(res, 1):
        print(f"{i}. {s}")
