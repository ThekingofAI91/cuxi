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
//   GET  /persona/legends           → {legends:[...], max_npcs, max_state_fields, max_only_fields}
//   GET  /persona/legend/{id}       → 完整存档（续玩用，含 state_fields/only_fields/state/only）
//   DELETE /persona/legend/{id}     → 删档
//
// 状态栏（作者定字段，AI 每轮更新）：
//   state_fields / state —— 通用，**在场每个人物各有一份**
//   only_fields  / only  —— 仅主角
//   每轮 end 事件带回最新的 state/only 与这一轮的 changed，前端据此刷新面板。
// ============================================================

const lgState = {
  view: 'lobby',        // lobby | setup | game
  saves: [],            // 存档摘要列表
  maxNpcs: 4,
  maxStateFields: 8,    // 服务端下发的上限，前端不写死
  maxOnlyFields: 6,
  style: 'classic',
  save: null,           // 当前局的完整存档
  loading: false,       // 正在生成（禁用输入）
  streamText: '',       // 本轮流式累积的叙述
  _streamEl: null,      // 正在流式渲染的容器
  changed: null,        // 这一轮真正变化的状态值（用来高亮）
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
    // 状态栏字段上限也由服务端给（两边各写一份迟早会漂）
    if (d.max_state_fields) lgState.maxStateFields = d.max_state_fields;
    if (d.max_only_fields) lgState.maxOnlyFields = d.max_only_fields;
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
  _lgClearFields();
  _lgSetStyle('classic');
  if (lgStatus) lgStatus.textContent = '';
  _lgShowStage('setup');
  lgWorld.focus();
}

// ---------- 状态栏字段（作者定义） ----------

function _lgClearFields() {
  if (lgStateFieldList) lgStateFieldList.innerHTML = '';
  if (lgOnlyFieldList) lgOnlyFieldList.innerHTML = '';
}

// 字段名的合法形态，与后端 legend_store._FIELD_NAME_RE 对齐。
// 名字会成为 AI 输出 JSON 里的 key，带空格/引号会把补丁解析带歪。
const LG_FIELD_NAME_BAD = /[\s"'`:{}\[\],，、]/;

function _lgAddFieldRow(container, max, seed) {
  if (!container) return;
  if (container.querySelectorAll('.lg-sfield-row').length >= max) {
    toast(`最多 ${max} 条（状态栏是给一眼扫的，太多就没人看了）`);
    return;
  }
  const row = document.createElement('div');
  row.className = 'lg-sfield-row';
  row.innerHTML = `
    <div class="lg-sfield-grid">
      <input class="lg-input lg-sfield-name" type="text" maxlength="12"
        placeholder="名称（如：好感度）" />
      <select class="lg-input lg-sfield-kind">
        <option value="number">数值</option>
        <option value="tag">词条</option>
      </select>
      <input class="lg-input lg-sfield-init" type="text" maxlength="24" placeholder="初值" />
    </div>
    <button class="lg-npc-del" type="button">移除此词条</button>`;

  const nameEl = row.querySelector('.lg-sfield-name');
  const kindEl = row.querySelector('.lg-sfield-kind');
  const initEl = row.querySelector('.lg-sfield-init');

  // 类型决定初值长什么样：数值给 0，词条留空（提示一个例子）
  const syncInit = () => {
    const isNum = kindEl.value === 'number';
    initEl.placeholder = isNum ? '初值（如：0）' : '初值（如：中立）';
    if (!initEl.value) initEl.value = isNum ? '0' : '';
  };
  kindEl.addEventListener('change', syncInit);

  if (seed) {
    nameEl.value = seed.name || '';
    kindEl.value = seed.kind || 'number';
    initEl.value = (seed.init === undefined || seed.init === null) ? '' : String(seed.init);
    syncInit();
  } else {
    syncInit();
  }
  row.querySelector('.lg-npc-del').addEventListener('click', () => row.remove());
  container.appendChild(row);
}

function _lgCollectFields(container) {
  if (!container) return [];
  return Array.from(container.querySelectorAll('.lg-sfield-row')).map((row) => {
    const kind = row.querySelector('.lg-sfield-kind').value;
    const raw = (row.querySelector('.lg-sfield-init').value || '').trim();
    let init;
    if (kind === 'number') {
      const n = Number(raw);
      init = (raw === '' || !Number.isFinite(n)) ? 0 : n;
    } else {
      init = raw;
    }
    return {
      name: (row.querySelector('.lg-sfield-name').value || '').trim(),
      kind,
      init,
    };
  }).filter((f) => f.name);
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
  const stateFields = _lgCollectFields(lgStateFieldList);
  const onlyFields = _lgCollectFields(lgOnlyFieldList);

  // 前端先做一遍明显错误的提示（后端的校验才是权威，这里只为省一次往返）
  if (!world) { toast('请填写世界观设定'); lgWorld.focus(); return; }
  if (world.length < 10) { toast('世界观太短了，至少写 10 个字'); lgWorld.focus(); return; }
  if (!heroName) { toast('请填写主角名字'); lgHeroName.focus(); return; }
  if (npcs.some((n) => n.name === heroName)) { toast('配角名字不能与主角相同'); return; }
  const names = npcs.map((n) => n.name);
  if (new Set(names).size !== names.length) { toast('配角名字有重复'); return; }
  const badField = stateFields.concat(onlyFields).find((f) => LG_FIELD_NAME_BAD.test(f.name));
  if (badField) { toast(`词条名「${badField.name}」不能带空格或标点`); return; }
  for (const [label, list] of [['通用词条', stateFields], ['主角词条', onlyFields]]) {
    const fs = list.map((f) => f.name);
    if (new Set(fs).size !== fs.length) { toast(`${label}里有重名`); return; }
  }

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
        state_fields: stateFields,
        only_fields: onlyFields,
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
  lgState.changed = null;      // 换局/续玩时不残留上一局的高亮
  _lgRenderStateBar();
}

// ---------- 状态栏渲染 ----------

// 在场人物：主角在前，其后是各配角（与后端 character_names_of 同一口径）
function _lgCharacterNames(s) {
  const out = [];
  const push = (n) => {
    const t = String(n || '').trim();
    if (t && out.indexOf(t) < 0) out.push(t);
  };
  push(s && s.protagonist_name);
  ((s && s.npcs) || []).forEach((n) => push(n && n.name));
  return out;
}

// 取一个字段的显示值；值缺失时回落到定义里的初值（与后端 _value_of 同一口径）
function _lgFieldValue(values, field) {
  const v = values ? values[field.name] : undefined;
  if (v === undefined || v === null) {
    return (field.init === undefined || field.init === null) ? '' : String(field.init);
  }
  if (field.kind === 'number') {
    const n = Number(v);
    return Number.isFinite(n) ? String(n) : String(v);
  }
  return String(v);
}

function _lgRenderStateBar() {
  if (!lgStateBar) return;
  const s = lgState.save;
  const common = (s && s.state_fields) || [];
  const only = (s && s.only_fields) || [];
  if (!s || (!common.length && !only.length)) {
    lgStateBar.hidden = true;
    lgStateBar.innerHTML = '';
    return;
  }
  const changed = lgState.changed || { state: {}, only: {} };
  const hitState = (who, name) => !!(changed.state && changed.state[who]
    && changed.state[who][name] !== undefined);
  const hitOnly = (name) => !!(changed.only && changed.only[name] !== undefined);

  const rows = [];
  if (common.length) {
    const chars = _lgCharacterNames(s).map((who) => {
      const vals = (s.state || {})[who] || {};
      const items = common.map((f) => {
        const cls = hitState(who, f.name) ? ' is-changed' : '';
        return `<span class="lg-sb-item${cls}">${escapeHtml(f.name)}` +
               `<b>${escapeHtml(_lgFieldValue(vals, f))}</b></span>`;
      }).join('');
      const heroCls = who === s.protagonist_name ? ' is-hero' : '';
      return `<span class="lg-sb-char"><span class="lg-sb-name${heroCls}">` +
             `${escapeHtml(who)}</span>${items}</span>`;
    }).join('');
    rows.push(`<div class="lg-sb-row"><span class="lg-sb-label">人物</span>` +
              `<span class="lg-sb-chars">${chars}</span></div>`);
  }
  if (only.length) {
    const items = only.map((f) => {
      const cls = hitOnly(f.name) ? ' is-changed' : '';
      return `<span class="lg-sb-item${cls}">${escapeHtml(f.name)}` +
             `<b>${escapeHtml(_lgFieldValue(s.only, f))}</b></span>`;
    }).join('');
    rows.push(`<div class="lg-sb-row"><span class="lg-sb-label">主角</span>${items}</div>`);
  }
  lgStateBar.innerHTML = rows.join('');
  lgStateBar.hidden = false;
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

// 行内切分：把一段正文里的「神情/动作」与「引号里的台词」各切成一个 span，
// 其余原样。三类内容分开着色，判据见 _lgRenderNarration 顶部说明。
function _lgInline(raw) {
  const s = String(raw || '');
  // 只匹配**成对闭合**的片段：不闭合就原样输出（宁可少着色，也不要吃掉正文）。
  // 引号要同时认全角与半角：真链路里模型多用半角 "…"（不是 “…”）。
  const re = /（[^（）]*）|「[^「」]*」|“[^“”]*”|"[^"]*"|‘[^‘’]*’/g;
  let out = '';
  let last = 0;
  let m;
  while ((m = re.exec(s)) !== null) {
    if (m.index > last) out += escapeHtml(s.slice(last, m.index));
    const seg = m[0];
    const cls = seg.charAt(0) === '（' ? 'lg-act' : 'lg-quote';
    out += `<span class="${cls}">${escapeHtml(seg.slice(1, -1))}</span>`;
    last = m.index + seg.length;
  }
  out += escapeHtml(s.slice(last));
  return out;
}

// 叙述渲染：一轮叙述按三类内容分开着色。
// 契约写在 framework/legend.py 的 system prompt 里，改一边要改两边：
//   人物说的话      `**名字**：台词` 独立成行（另有行内引号）
//   人物神情与动作  行内全角括号（…）
//   环境与背景      其余正文，不加标记
//
// ★一行 = 一个段落。以前是把连续几行 join('') 合成一个 <p>，那会把模型写的
//   「\n\n」段间隔吃掉、几段黏成一段（真存档里每段都是一行），所以改成逐行成段。
function _lgRenderNarration(text) {
  const lines = String(text || '').split(/\n+/);
  let html = '';
  lines.forEach((raw) => {
    const line = raw.trim();
    if (!line) return;
    const m = line.match(/^\*\*(.+?)\*\*\s*[：:]\s*([\s\S]*)$/);
    if (m) {
      html += `<div class="lg-dialogue"><span class="lg-speaker">${escapeHtml(m[1])}</span>` +
              `<span class="lg-line">${_lgInline(m[2])}</span></div>`;
    } else {
      html += `<p class="lg-para">${_lgInline(line)}</p>`;
    }
  });
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
      // 状态栏：后端带回整份最新值 + 这一轮真正改过的部分
      if (data.state) lgState.save.state = data.state;
      if (data.only) lgState.save.only = data.only;
      lgState.changed = data.changed || null;
      _lgRenderStateBar();
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
if (lgAddStateFieldBtn) {
  lgAddStateFieldBtn.addEventListener('click',
    () => _lgAddFieldRow(lgStateFieldList, lgState.maxStateFields));
}
if (lgAddOnlyFieldBtn) {
  lgAddOnlyFieldBtn.addEventListener('click',
    () => _lgAddFieldRow(lgOnlyFieldList, lgState.maxOnlyFields));
}
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
