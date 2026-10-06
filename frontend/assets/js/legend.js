// ============================================================
// 传奇（剧情模式）
//
// 与「问道 / 会心 / 争鸣」同级的一种玩法：用户给出世界观、主角、配角三段设定，
// 然后**自己扮演主角**往下推剧情；叙述与所有配角的言行由同一个模型扮演。
//
// 与另外两条线的边界（别混）：
//   · 争鸣 = 选**已有**角色就议题交锋，用户是提问者/点将者，终点是纪要。
//   · 自建角色 = 建一个**对话对象**，进首页列表，可以单独找 TA 聊。
//   · 传奇 = 角色用户现场创作、用户下场演主角、剧情无限、不进任何角色列表。
//
// 后端契约（src/api/routes.py 末尾「传奇」段）：
//   POST /persona/legend/create      → {id, title, protagonist, npcs}
//   POST /persona/legend/{id}/act   → SSE: start / token / end / error
//   GET  /persona/legends           → {legends:[...], max_npcs}
//   GET  /persona/legend/{id}       → 完整存档（续玩用）
//   DELETE /persona/legend/{id}     → 删档
// ============================================================

const lgState = {
  view: 'lobby',        // lobby | setup | game
  saves: [],            // 存档摘要列表
  maxNpcs: 4,
  style: 'classic',
  save: null,           // 当前局的完整存档
  loading: false,       // 正在生成（禁用输入）
  streamText: '',       // 本轮流式累积的叙述
  _streamEl: null,      // 正在流式渲染的容器
};

// 叙事风格的中文名（提交给后端的是 key）
const LG_STYLE_LABELS = { classic: '正剧', light: '轻松', dark: '肃杀' };


// ============================================================
// 视图切换
// ============================================================

function openLegend() {
  closeSidebar();
  legendView.hidden = false;
  _lgShowStage('lobby');
  loadLegendSaves();
}

function closeLegend() {
  legendView.hidden = true;
}

function _lgShowStage(stage) {
  lgState.view = stage;
  lgLobby.hidden = stage !== 'lobby';
  lgSetup.hidden = stage !== 'setup';
  lgGame.hidden = stage !== 'game';
}


// ============================================================
// 阶段一：存档列表
// ============================================================

async function loadLegendSaves() {
  if (lgMaxNpcEl) lgMaxNpcEl.textContent = String(lgState.maxNpcs);
  try {
    const r = await fetch('/persona/legends');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    lgState.saves = d.legends || [];
    if (d.max_npcs) {
      lgState.maxNpcs = d.max_npcs;
      if (lgMaxNpcEl) lgMaxNpcEl.textContent = String(d.max_npcs);
    }
  } catch (e) {
    lgState.saves = [];
  }
  renderLegendSaves();
}

function renderLegendSaves() {
  if (!lgSaves) return;
  lgSaves.innerHTML = '';
  const list = lgState.saves;
  if (!list.length) {
    const empty = document.createElement('div');
    empty.className = 'lg-empty';
    empty.innerHTML = '还没有进行中的故事。<br>新建一局，写下你想要的世界。';
    lgSaves.appendChild(empty);
    return;
  }
  list.forEach((s) => {
    const card = document.createElement('div');
    card.className = 'lg-save-card';
    const when = s.updated_at
      ? new Date(s.updated_at * 1000).toLocaleDateString('zh-CN')
      : '';
    card.innerHTML = `
      <div class="lg-save-main">
        <div class="lg-save-title">${escapeHtml(s.title || '未命名')}</div>
        <div class="lg-save-meta">${s.turns} 轮 · ${s.npc_count} 位配角${when ? ' · ' + when : ''}</div>
      </div>
      <div class="lg-save-ops">
        <button class="lg-btn small primary" data-op="open" type="button">继续</button>
        <button class="lg-btn small ghost danger" data-op="del" type="button">删除</button>
      </div>`;
    card.querySelector('[data-op="open"]').addEventListener('click', () => {
      openLegendSave(s.id);
    });
    card.querySelector('[data-op="del"]').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteLegendSave(s.id, s.title);
    });
    lgSaves.appendChild(card);
  });
}

async function deleteLegendSave(id, title) {
  if (!window.confirm(`删除《${title || '这个存档'}》？剧情会一起删掉，无法恢复。`)) return;
  try {
    const r = await fetch(`/persona/legend/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    toast('已删除');
    loadLegendSaves();
  } catch (e) {
    toast('删除失败：' + (e.message || e));
  }
}


// ============================================================
// 阶段二：创建设定
// ============================================================

function openLegendSetup() {
  lgWorld.value = '';
  lgHeroName.value = '';
  lgHeroDesc.value = '';
  lgOpening.value = '';
  lgNpcs.innerHTML = '';
  _lgAddNpcRow();   // 默认给一个配角位，省得用户找不到入口
  _lgSetStyle('classic');
  if (lgStatus) lgStatus.textContent = '';
  _lgShowStage('setup');
  lgWorld.focus();
}

function _lgSetStyle(style) {
  lgState.style = style;
  if (!lgStyles) return;
  lgStyles.querySelectorAll('.lg-style-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.style === style);
  });
}

function _lgAddNpcRow() {
  const rows = lgNpcs.querySelectorAll('.lg-npc-row');
  if (rows.length >= lgState.maxNpcs) {
    toast(`配角最多 ${lgState.maxNpcs} 位（人太多每人的戏份会被摊薄）`);
    return;
  }
  const row = document.createElement('div');
  row.className = 'lg-npc-row';
  row.innerHTML = `
    <div class="lg-npc-grid">
      <input class="lg-input lg-npc-name" type="text" maxlength="20" placeholder="名字" />
      <input class="lg-input lg-npc-role" type="text" maxlength="20" placeholder="身份（如：酒肆老板娘）" />
    </div>
    <textarea class="lg-textarea lg-npc-persona" maxlength="400"
      placeholder="性格、口吻、与主角的关系、知道些什么。越具体，TA 越像个活人。"></textarea>
    <button class="lg-npc-del" type="button">移除此配角</button>`;
  row.querySelector('.lg-npc-del').addEventListener('click', () => row.remove());
  lgNpcs.appendChild(row);
}

function _lgCollectNpcs() {
  return Array.from(lgNpcs.querySelectorAll('.lg-npc-row')).map((row) => ({
    name: (row.querySelector('.lg-npc-name').value || '').trim(),
    role: (row.querySelector('.lg-npc-role').value || '').trim(),
    persona: (row.querySelector('.lg-npc-persona').value || '').trim(),
  })).filter((n) => n.name);
}

async function createLegend() {
  const world = (lgWorld.value || '').trim();
  const heroName = (lgHeroName.value || '').trim();
  const npcs = _lgCollectNpcs();

  // 前端先做一遍明显错误的提示（后端的校验才是权威，这里只为省一次往返）
  if (!world) { toast('请填写世界观设定'); lgWorld.focus(); return; }
  if (world.length < 10) { toast('世界观太短了，至少写 10 个字'); lgWorld.focus(); return; }
  if (!heroName) { toast('请填写主角名字'); lgHeroName.focus(); return; }
  if (npcs.some((n) => n.name === heroName)) { toast('配角名字不能与主角相同'); return; }
  const names = npcs.map((n) => n.name);
  if (new Set(names).size !== names.length) { toast('配角名字有重复'); return; }

  lgCreateBtn.disabled = true;
  if (lgStatus) lgStatus.textContent = '正在开篇……';
  try {
    const r = await fetch('/persona/legend/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        world,
        protagonist_name: heroName,
        protagonist_desc: (lgHeroDesc.value || '').trim(),
        npcs,
        opening: (lgOpening.value || '').trim(),
        style: lgState.style,
      }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || ('HTTP ' + r.status));
    if (lgStatus) lgStatus.textContent = '';
    await startLegendGame(d.id, true);
  } catch (e) {
    if (lgStatus) lgStatus.textContent = '';
    toast('创建失败：' + (e.message || e));
  } finally {
    lgCreateBtn.disabled = false;
  }
}


// ============================================================
// 阶段三：游戏
// ============================================================

async function openLegendSave(id) {
  try {
    const r = await fetch(`/persona/legend/${encodeURIComponent(id)}`);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const save = await r.json();
    lgState.save = save;
    _lgEnterGame();
    _lgRenderHistory();
    // 已有剧情就滚到底；没有剧情（刚建好还没开局）则触发开场
    if (!save.turns || !save.turns.length) {
      await _lgRequest(null);
    } else {
      _lgScroll();
    }
  } catch (e) {
    toast('读取存档失败：' + (e.message || e));
  }
}

async function startLegendGame(id, fresh) {
  const r = await fetch(`/persona/legend/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  lgState.save = await r.json();
  _lgEnterGame();
  lgTranscript.innerHTML = '';
  await _lgRequest(null);   // 开局
}

function _lgEnterGame() {
  _lgShowStage('game');
  const s = lgState.save;
  if (lgSceneBar && s) {
    const npcs = (s.npcs || []).map((n) => n.name).filter(Boolean);
    lgSceneBar.innerHTML =
      `<span class="lg-scene-hero">${escapeHtml(s.protagonist_name || '')}</span>` +
      (npcs.length ? `<span class="lg-scene-npcs">配角：${escapeHtml(npcs.join('、'))}</span>` : '') +
      `<span class="lg-scene-style">${escapeHtml(LG_STYLE_LABELS[s.style] || '正剧')}</span>`;
  }
  _lgSetBusy(false);
  if (lgAction) lgAction.value = '';
}

function _lgRenderHistory() {
  lgTranscript.innerHTML = '';
  const turns = (lgState.save && lgState.save.turns) || [];
  turns.forEach((t) => {
    if (t.role === 'user') {
      _lgAppendAction(t.content);
    } else if (t.role === 'narrator') {
      _lgAppendNarration(t.content);
    }
  });
}

function _lgAppendAction(text) {
  const div = document.createElement('div');
  div.className = 'lg-action-echo';
  div.innerHTML = `<span class="lg-action-tag">你</span>${escapeHtml(text)}`;
  lgTranscript.appendChild(div);
  _lgScroll();
}

// 叙述渲染：把 `**名字**：台词` 的行单独高亮，其余按段落输出。
// 这个格式是 framework/legend.py 的 system prompt 里约定死的，改一边要改两边。
function _lgRenderNarration(text) {
  const lines = String(text || '').split(/\n+/);
  let html = '';
  let para = [];
  const flushPara = () => {
    if (para.length) {
      html += `<p class="lg-para">${escapeHtml(para.join(''))}</p>`;
      para = [];
    }
  };
  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) { flushPara(); return; }
    const m = line.match(/^\*\*(.+?)\*\*\s*[：:]\s*(.*)$/);
    if (m) {
      flushPara();
      html += `<div class="lg-dialogue"><span class="lg-speaker">${escapeHtml(m[1])}</span>` +
              `<span class="lg-line">${escapeHtml(m[2])}</span></div>`;
    } else {
      para.push(line);
    }
  });
  flushPara();
  return html || `<p class="lg-para">${escapeHtml(text || '')}</p>`;
}

function _lgAppendNarration(text) {
  const div = document.createElement('div');
  div.className = 'lg-narration';
  div.innerHTML = _lgRenderNarration(text);
  lgTranscript.appendChild(div);
  _lgScroll();
}

function _lgScroll() {
  if (lgTranscript) lgTranscript.scrollTop = lgTranscript.scrollHeight;
}

function _lgSetBusy(busy) {
  lgState.loading = busy;
  if (lgSendBtn) lgSendBtn.disabled = busy;
  if (lgAction) lgAction.disabled = busy;
  if (lgSendBtn) lgSendBtn.textContent = busy ? '…' : '行动';
}

// 发一次请求（action 为 null = 开局）
async function _lgRequest(action) {
  const s = lgState.save;
  if (!s || lgState.loading) return;

  if (action) _lgAppendAction(action);
  _lgSetBusy(true);

  // 流式容器：先插一个空壳，token 来了往里填
  const holder = document.createElement('div');
  holder.className = 'lg-narration lg-streaming';
  holder.innerHTML = '<p class="lg-para lg-typing">……</p>';
  lgTranscript.appendChild(holder);
  _lgScroll();
  lgState.streamText = '';

  try {
    const resp = await fetch(`/persona/legend/${encodeURIComponent(s.id)}/act`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: action || '', history: [] }),
    });
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      throw new Error(d.detail || ('HTTP ' + resp.status));
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const line of parts) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch (e) { continue; }
        _lgHandleEvent(data, holder);
      }
    }
    if (buffer.trim().startsWith('data: ')) {
      try { _lgHandleEvent(JSON.parse(buffer.trim().slice(6)), holder); } catch (e) { /* ignore */ }
    }
  } catch (err) {
    holder.innerHTML = `<p class="lg-para lg-error">${escapeHtml(err.message || String(err))}</p>`;
  } finally {
    holder.classList.remove('lg-streaming');
    // 流式期间渲染的是「半截」，结束后用最终文本重渲染一次，保证 markdown 完整
    if (lgState.streamText) {
      holder.innerHTML = _lgRenderNarration(lgState.streamText);
    }
    lgState.streamText = '';
    _lgSetBusy(false);
    _lgScroll();
    if (lgAction && !lgAction.disabled) lgAction.focus();
  }
}

function _lgHandleEvent(data, holder) {
  if (!data || !data.type) return;
  if (data.type === 'token') {
    lgState.streamText += (data.content || '');
    holder.innerHTML = _lgRenderNarration(lgState.streamText);
    _lgScroll();
    return;
  }
  if (data.type === 'end') {
    if (data.legend_id && lgState.save && lgState.save.id === data.legend_id) {
      // 后端已落盘，本地同步轮数以保持场景栏与列表一致
      lgState.save.turn_count = data.turns || lgState.save.turn_count;
    }
    return;
  }
  if (data.type === 'error') {
    holder.innerHTML = `<p class="lg-para lg-error">${escapeHtml(data.content || '出错了')}</p>`;
    lgState.streamText = '';
  }
}

function sendLegendAction() {
  const text = (lgAction.value || '').trim();
  if (!text) { toast('先写点什么，替主角做个决定'); return; }
  if (lgState.loading) return;
  lgAction.value = '';
  _lgRequest(text);
}


// ============================================================
// 事件绑定
// ============================================================

if (lgBackBtn) {
  lgBackBtn.addEventListener('click', () => {
    // 从设定页返回 = 回存档列表；从游戏里返回 = 也回列表（存档已落盘，不丢进度）
    if (lgState.view === 'lobby') {
      closeLegend();
    } else {
      _lgShowStage('lobby');
      loadLegendSaves();
    }
  });
}
if (lgNewBtn) lgNewBtn.addEventListener('click', openLegendSetup);
if (lgAddNpcBtn) lgAddNpcBtn.addEventListener('click', _lgAddNpcRow);
if (lgCreateBtn) lgCreateBtn.addEventListener('click', createLegend);
if (lgSetupCancelBtn) {
  lgSetupCancelBtn.addEventListener('click', () => {
    _lgShowStage('lobby');
    loadLegendSaves();
  });
}
if (lgStyles) {
  lgStyles.querySelectorAll('.lg-style-btn').forEach((btn) => {
    btn.addEventListener('click', () => _lgSetStyle(btn.dataset.style));
  });
}
if (lgSendBtn) lgSendBtn.addEventListener('click', sendLegendAction);
if (lgAction) {
  lgAction.addEventListener('keydown', (e) => {
    // 回车发送、Shift+回车换行（与对话输入框的手感保持一致）
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendLegendAction();
    }
  });
}
if (legendView) {
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !legendView.hidden) {
      if (lgState.view === 'lobby') closeLegend();
      else { _lgShowStage('lobby'); loadLegendSaves(); }
    }
  });
}
