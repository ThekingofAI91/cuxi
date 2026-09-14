"""
persona_builder.py — 把「名字 + 用户背景 + 网络素材」合成为可对话人设

两条路径：
1. 用户没写 role_prompt → _generate_both：让 LLM 同时产出
   - role_prompt：第一人称人设（口吻/立场/知识边界/开场风格），对标荣格角色的定义质量
   - background：一段结构化背景知识库（markdown），用于灌入 ChromaDB 供检索
2. 用户写了 role_prompt → 保留用户的人设，仅在网络素材存在时用 _merge_background
   把公开资料合并进 background（绝不篡改用户精心写好的角色设定）

所有 LLM 输出都用 llm_json.extract_json 容错解析。
"""

from __future__ import annotations

from src.core.llm import get_chat_llm
from src.core.llm_json import extract_json
from src.core.config import settings
from src.web_search import web_research

_SYSTEM_BOTH = """你是"可对话 AI 角色"的人设设计师。用户要创建一个能与大家对话的名人/人物仿真角色。
你需要根据人物名字、用户提供的背景，以及（可选的）网络公开资料，产出两部分内容：

1) role_prompt（角色人设提示词）：用第一人称写，决定这个角色"怎么说话、持什么立场、知道什么、边界在哪"。
   要求：
   - 明确"你是谁"（姓名、生卒/年代、身份），用第一人称"我"。
   - 说话风格具体可执行：是否用动作描写（*动作*）、语气温和/犀利/幽默、是否引用自己的代表作或名言。
   - 知识边界：你的知识截止于哪一年、基于哪些著作/公开言论；超出的诚实说明"我没有研究过"。
   - 核心视角：从角色的专业/人生态度理解问题。
   - 开场白：自然开始，不要反复声明"我是AI/助手"（页面已有声明），除非用户问起。
   - 直接给结论、给可操作建议，不要只绕圈子。
   - 控制在 300-600 字，信息密度高、可直接当 system prompt 用。

2) background（背景知识库）：一段结构化的 markdown，汇总这个角色的关键事实——
   生平时间线、代表成就/著作、核心思想/观点、广为流传的名言、典型争议或趣闻。
   用于灌入检索库，让回答有事实支撑。网络资料与用户背景冲突时，以用户背景为准，
   网络资料作为补充，并标注来源年份。控制在 400-900 字。

3) card（酒馆式角色卡）：决定这个角色"像不像本人"，而不是像 AI 助手。五个字段：
   - first_mes：开场白。角色对来访者说的第一句话，一句即可，要立刻显出性格。
   - mes_example：2-3 轮示例对话，用 {{user}} / {{char}} 标记说话人。
     要展示这个人的语气、口头禅、句子长短、如何给出结论——不是展示知识，是展示"说话的样子"。
   - scenario：一两句话的场景设定（在哪儿、什么情境下谈话）。
   - personality：3-5 条性格特质，每条一行，以 - 开头。
   - lorebook：4-6 条关键词词条，形如 {"keyword": "...", "content": "..."}。
     keyword 必须写成用户真的会说出口的词（如"知行合一""天坑专业""性压抑"）；
     content 用第一人称写这个角色自己对该词的说法或界定，不要写成百科词条。

只输出 JSON，形如：
{"role_prompt": "...", "background": "...",
 "first_mes": "...", "mes_example": "...", "scenario": "...",
 "personality": "...", "lorebook": [{"keyword": "...", "content": "..."}]}
不要输出任何解释性文字。"""

_SYSTEM_MERGE = """你是"人物背景资料整理师 + 角色卡设计师"。用户已经写好角色人设（role_prompt）。
你需要产出两部分：

1) background：把【用户背景】与【网络公开资料】整理成结构化 markdown 背景知识库，供检索使用。
   - 合并用户背景与网络资料，用户背景优先级更高；冲突时以用户背景为准。
   - 结构：生平时间线、代表成就/著作、核心思想、流传名言、典型争议/趣闻。
   - 网络资料标注大致年份或"据公开资料"。
   - 控制在 400-900 字。

2) card（酒馆式角色卡）：严格依据用户写好的 role_prompt 提炼，不得引入与之冲突的设定。五个字段：
   - first_mes：开场白，一句，说话方式必须符合人设里的规定。
   - mes_example：2-3 轮示例对话，用 {{user}} / {{char}} 标记，
     模范人设里描述的语气、口头禅、句子长短与给结论的方式。
   - scenario：一两句话的场景设定。
   - personality：3-5 条性格特质，每条一行，以 - 开头。
   - lorebook：4-6 条关键词词条 {"keyword": "...", "content": "..."}。
     keyword 写成用户真的会说出口的词；content 用第一人称写角色自己的说法，与人设保持一致。

只输出 JSON：
{"background": "...", "first_mes": "...", "mes_example": "...", "scenario": "...",
 "personality": "...", "lorebook": [{"keyword": "...", "content": "..."}]}
不要解释。"""


def _extract_card(obj: dict) -> dict:
    """
    从 LLM 返回的 JSON 里取出酒馆式角色卡的五个字段，并清洗世界书格式。

    世界书只保留 keyword/content 都非空的条目——残缺词条注入 prompt 只会添乱。
    """
    lorebook: list[dict] = []
    raw_lore = obj.get("lorebook") or []
    if isinstance(raw_lore, list):
        for item in raw_lore:
            if not isinstance(item, dict):
                continue
            kw = (item.get("keyword") or "").strip()
            content = (item.get("content") or "").strip()
            if kw and content:
                lorebook.append({"keyword": kw, "content": content})
    return {
        "first_mes": (obj.get("first_mes") or "").strip(),
        "mes_example": (obj.get("mes_example") or "").strip(),
        "scenario": (obj.get("scenario") or "").strip(),
        "personality": (obj.get("personality") or "").strip(),
        "lorebook": lorebook,
    }


async def _generate_both(name: str, user_background: str, web_text: str) -> tuple[str, str, dict]:
    parts = [f"人物名字：{name}"]
    if user_background and user_background.strip():
        parts.append(f"用户提供的背景：\n{user_background.strip()}")
    if web_text and web_text.strip():
        parts.append(f"网络公开资料（仅供参考，可能不全/有误）：\n{web_text.strip()}")
    user_msg = "\n\n".join(parts)
    llm = get_chat_llm(temperature=0.7, max_tokens=2500)
    resp = await llm.ainvoke([("system", _SYSTEM_BOTH), ("user", user_msg)])
    obj = extract_json(resp.content)
    role_prompt = (obj.get("role_prompt") or "").strip()
    background = (obj.get("background") or "").strip()
    return role_prompt, background, _extract_card(obj)


async def _merge_background(
    name: str, user_background: str, web_text: str, role_prompt: str = ""
) -> tuple[str, dict]:
    """用户自带人设时：只整理背景，同时依据该人设提炼角色卡（同一次调用，不额外耗时）"""
    parts = [f"人物名字：{name}"]
    if role_prompt and role_prompt.strip():
        parts.append(f"用户写好的角色人设（角色卡必须严格据此提炼）：\n{role_prompt.strip()}")
    if user_background and user_background.strip():
        parts.append(f"用户提供的背景：\n{user_background.strip()}")
    if web_text and web_text.strip():
        parts.append(f"网络公开资料（仅供参考）：\n{web_text.strip()}")
    user_msg = "\n\n".join(parts)
    llm = get_chat_llm(temperature=0.4, max_tokens=2000)
    resp = await llm.ainvoke([("system", _SYSTEM_MERGE), ("user", user_msg)])
    obj = extract_json(resp.content)
    return (obj.get("background") or "").strip(), _extract_card(obj)


async def synthesize_persona(
    name: str,
    user_background: str,
    web_text: str,
    provided_role_prompt: str | None = None,
) -> tuple[str, str, dict]:
    """
    返回 (role_prompt, background, card)。

    - 用户提供 role_prompt：保留之，仅在有网络素材时整理 background 并提炼角色卡
      （同一次调用完成，不额外耗时）；无网络素材时角色卡留空。
    - 用户未提供：LLM 一并生成 role_prompt、background 与角色卡。

    card 为酒馆式角色卡：first_mes / mes_example / scenario / personality / lorebook。
    """
    if provided_role_prompt is not None:
        if web_text and web_text.strip():
            background, card = await _merge_background(
                name, user_background or "", web_text, provided_role_prompt,
            )
        else:
            background = (user_background or "").strip()
            card = _extract_card({})
        return provided_role_prompt, background, card

    role_prompt, background, card = await _generate_both(
        name, user_background or "", web_text or "",
    )
    # 兜底：LLM 没给出 role_prompt 时，用最小可用人设
    if not role_prompt:
        role_prompt = (
            f"你是{name}。请以{name}的身份、口吻和立场与用户对话，"
            f"用第一人称，保持角色一致性。基于你已知的公开信息与用户提供的背景回答，"
            f"超出知识范围时诚实说明。"
        ).strip()
    if not background:
        background = (user_background or name).strip()
    return role_prompt, background, card


async def build_persona(
    name: str,
    user_background: str,
    use_web_search: bool,
    provided_role_prompt: str | None = None,
) -> tuple[str, str, list[str], dict]:
    """
    端到端构建人设。返回 (role_prompt, background, sources, card)。
    - sources：实际用到的网络片段（空列表表示未启用/未搜到，调用方据此提示用户）。
    - card：酒馆式角色卡（开场白 / 示例对话 / 场景 / 性格 / 世界书）。
    """
    sources: list[str] = []
    web_text = ""
    if use_web_search and settings.web_search_enabled:
        try:
            snippets = await web_research(
                name,
                max_results=settings.web_search_max_results,
                timeout=settings.web_search_timeout,
                provider=settings.web_search_provider,
            )
            sources = snippets
            if snippets:
                web_text = "\n".join(f"- {s}" for s in snippets)
        except Exception as e:
            print(f"[persona_builder] web search failed: {e}")
    role_prompt, background, card = await synthesize_persona(
        name, user_background or "", web_text, provided_role_prompt,
    )
    return role_prompt, background, sources, card
