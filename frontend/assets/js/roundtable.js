// ============================================================
// 争鸣（圆桌会议）
//
// 与「问道 / 会心」同级的一种玩法：请两三位智者同桌，就一个议题各自判断要不要
// 开口。我来主持时先看完整请缨，再亲自点将或结束；代班主持按请缨紧迫度安排。
// 结尾由主持人整理各方立场、分歧与适用条件，不判胜负。
// ============================================================
const RT_SAVED_SESSION_KEY = 'chuxi.roundtable.last-session';
const rtState = {
  selected: new Set(),   // 已选与会者 id
  rounds: 3,             // 全场交锋发言次数，不含每人的开场
  picker: 'user',        // 谁主持：user = 用户点将 / agent = 主持人代班
  loading: false,
  sessionId: null,       // 本场会议 id（点将端点需要它）
  _current: null,        // 当前正在流式渲染的发言卡 { card, content }
  _bar: null,            // 本轮公告条（谁想说话 / 最后谁发言）
  _askResolve: null,
  _recoveryResolve: null,
  _speakers: [],
  _runId: 0,             // 离开或重开后，不再接收旧会议的流与点将回执
  _summaryEl: null,
  _stage: null,          // 围坐舞台：角色 id -> 座位元素
  _stageHost: null,      // 主持位元素
};

function renderRoundtableChips() {
  const chars = state.characters || [];
  const full = rtState.selected.size >= RT_MAX;
  rtChips.innerHTML = '';
  chars.forEach(ch => {
    const chip = document.createElement('button');
    chip.type = 'button';
    const active = rtState.selected.has(ch.id);
    // 选满后，未选中的人物禁用，避免超额
    const disabled = full && !active;
    chip.className = 'rt-chip' + (active ? ' active' : '') + (disabled ? ' disabled' : '');
    chip.dataset.charId = ch.id;
    if (disabled) chip.setAttribute('disabled', 'true');
    chip.innerHTML = `
      <span class="rt-chip-avatar">${escapeHtml(nameMark(ch.name))}</span>
      <span>${escapeHtml(ch.name)}</span>
      <span class="rt-chip-check">✓</span>`;
    chip.addEventListener('click', () => toggleRtChip(ch.id));
    rtChips.appendChild(chip);
  });
  updateRtStartBtn();
}

function toggleRtChip(id) {
  if (rtState.selected.has(id)) {
    rtState.selected.delete(id);
  } else {
    if (rtState.selected.size >= RT_MAX) {
      toast(`最多只能选择 ${RT_MAX} 位与会者（人太多会削弱交锋感）`);
      return;
    }
    rtState.selected.add(id);
  }
  rtChips.querySelectorAll('.rt-chip').forEach(c =>
    c.classList.toggle('active', rtState.selected.has(c.dataset.charId)));
  updateRtStartBtn();
}

function updateRtStartBtn() {
  const n = rtState.selected.size;
  rtStartBtn.disabled = rtState.loading || n < 2 || n > RT_MAX || !rtTopic.value.trim();
}

function _rtSavedSession() {
  try {
    const saved = JSON.parse(localStorage.getItem(RT_SAVED_SESSION_KEY) || 'null');
    return saved && typeof saved.session_id === 'string' ? saved : null;
  } catch (e) { return null; }
}

function _rtRememberSession(sessionId, topic, status = 'running') {
  if (!sessionId) return;
  try {
    localStorage.setItem(RT_SAVED_SESSION_KEY, JSON.stringify({ session_id: sessionId, topic: topic || '', status }));
  } catch (e) { /* 本地存储不可用时，当前页面仍可继续。 */ }
  _rtUpdateResumeEntry();
}

function _rtUpdateResumeEntry() {
  const entry = document.getElementById('rtResumeEntry');
  const label = document.getElementById('rtResumeLabel');
  const button = document.getElementById('rtResumeBtn');
  const saved = _rtSavedSession();
  if (entry) entry.hidden = !saved;
  if (label && saved) label.textContent = saved.topic ? `上次会议：${saved.topic}` : '上次会议已保留';
  if (button) button.textContent = saved && saved.status === 'completed' ? '查看上次会议' : '继续会议';
}

function openRoundtable() {
  closeSidebar();
  rtState._runId += 1;
  rtState.loading = false;
  rtTopic.disabled = false;
  // 先显示设置与续会入口；用户点击续会后，以服务端快照重建现场。
  rtState.sessionId = null;
  rtState._current = null;
  rtState._bar = null;
  rtState._summaryEl = null;
  _rtAskResolve(null);
  _rtRecoveryResolve(null);
  _rtStageReset();
  if (rtTranscript) rtTranscript.innerHTML = '';
  if (rtSummary) { rtSummary.hidden = true; rtSummary.innerHTML = ''; }
  _rtUpdateResumeEntry();

  // 首次进入：若人物列表为空则补拉一次
  if (!state.characters || state.characters.length === 0) {
    loadCharacters().then(() => {
      // 默认勾选前两位
      if (rtState.selected.size === 0 && state.characters.length >= 2) {
        rtState.selected.add(state.characters[0].id);
        rtState.selected.add(state.characters[1].id);
      }
      renderRoundtableChips();
    });
  } else {
    if (rtState.selected.size === 0 && state.characters.length >= 2) {
      rtState.selected.add(state.characters[0].id);
      rtState.selected.add(state.characters[1].id);
    }
    renderRoundtableChips();
  }
  roundtableView.hidden = false;
}

function closeRoundtable() {
  rtState._runId += 1;
  _rtAskResolve(null);
  _rtRecoveryResolve(null);
  rtState.loading = false;
  rtTopic.disabled = false;
  updateRtStartBtn();
  _rtStageReset();
  roundtableView.hidden = true;
}

function _rtRoundLabel(phase, round, total) {
  if (phase === 'opening') return '开场陈述';
  return `第 ${round}/${total} 次交锋`;
}

function _rtAppendDivider(phase, round, total) {
  const div = document.createElement('div');
  div.className = 'rt-round-divider';
  div.textContent = _rtRoundLabel(phase, round, total);
  rtTranscript.appendChild(div);
  rtState._bar = null;   // 换轮：公告条重建
  _rtScroll();
}

// ---- 本轮公告条：把"谁想说话"这个过程显性化 ----
function _rtBar() {
  if (!rtState._bar || !rtState._bar.isConnected) {
    const bar = document.createElement('div');
    bar.className = 'rt-bar';
    rtTranscript.appendChild(bar);
    rtState._bar = bar;
  }
  return rtState._bar;
}

function _rtBarText(text) {
  const bar = _rtBar();
  bar.innerHTML = `<span class="rt-bar-text">${escapeHtml(text)}</span>`;
  _rtScroll();
}

function _rtBarChip(name, reason) {
  const bar = _rtBar();
  let chips = bar.querySelector('.rt-bar-chips');
  if (!chips) {
    chips = document.createElement('div');
    chips.className = 'rt-bar-chips';
    bar.appendChild(chips);
  }
  const el = document.createElement('span');
  el.className = 'rt-bar-chip';
  el.innerHTML = `<b>${escapeHtml(name)}</b>${reason ? ' · ' + escapeHtml(reason) : ''}`;
  chips.appendChild(el);
  _rtScroll();
}

function _rtNotice(text, kind) {
  const div = document.createElement('div');
  div.className = 'rt-notice' + (kind ? ' rt-notice-' + kind : '');
  div.textContent = text;
  rtTranscript.appendChild(div);
  _rtScroll();
}

function _rtScroll() {
  if (rtTranscript) rtTranscript.scrollTop = rtTranscript.scrollHeight;
}

function _rtFinalizeCurrent() {
  if (rtState._current) {
    rtState._current.card.classList.remove('thinking');
    rtState._current = null;
  }
}

function _rtCreateSpeechCard(data) {
  const card = document.createElement('div');
  card.className = 'rt-speaker theme-' + (data.theme || 'original') + ' thinking';
  const count = data.quota ? `${data.speeches || 1}/${data.quota}` : '';
  card.innerHTML = `
    <div class="rt-speaker-avatar">${escapeHtml(nameMark(data.name))}</div>
    <div class="rt-speaker-body">
      <div class="rt-speaker-name">${escapeHtml(data.name)} <span class="ai-tag" title="本条内容由人工智能生成（AI-generated content）">AI 生成</span> <span class="rt-speaker-tag">正在发言…</span>${count ? `<span class="rt-speaker-count">${escapeHtml(count)}</span>` : ''}</div>
      <div class="rt-speaker-text"></div>
    </div>`;
  rtTranscript.appendChild(card);
  return { card, content: card.querySelector('.rt-speaker-text'), nameTag: card.querySelector('.rt-speaker-tag'), text: '' };
}

function _rtApplySnapshot(snapshot) {
  rtState.sessionId = snapshot.session_id || rtState.sessionId;
  rtState._speakers = Array.isArray(snapshot.speakers) ? snapshot.speakers : rtState._speakers;
  rtState.picker = snapshot.picker || rtState.picker;
  rtState.rounds = Number(snapshot.rounds) || rtState.rounds;
  rtTopic.value = snapshot.topic || '';
  rtState.selected = new Set(rtState._speakers.map(s => s.id));
  renderRoundtableChips();
  if (rtRounds) rtRounds.querySelectorAll('.rt-round-btn').forEach(button =>
    button.classList.toggle('active', Number(button.dataset.rounds) === rtState.rounds));
  if (rtPicker) rtPicker.querySelectorAll('.rt-round-btn').forEach(button =>
    button.classList.toggle('active', button.dataset.picker === rtState.picker));
  _rtRememberSession(rtState.sessionId, snapshot.topic, snapshot.status);
  _rtStageBuild(rtState._speakers, rtState.picker, snapshot.topic);
  rtTranscript.innerHTML = '';
  rtState._current = null;
  rtState._bar = null;
  const counts = {};
  let previousGroup = '';
  for (const turn of snapshot.transcript || []) {
    const id = turn.character_id || turn.speaker_id;
    const speaker = rtState._speakers.find(s => s.id === id) || {};
    const kind = turn.kind || turn.phase || (Number(turn.round) ? 'rebuttal' : 'opening');
    const group = `${kind}:${turn.round || 0}`;
    if (group !== previousGroup) _rtAppendDivider(kind, turn.round || 0, rtState.rounds);
    previousGroup = group;
    counts[id] = (counts[id] || 0) + 1;
    const view = _rtCreateSpeechCard({ ...speaker, name: turn.name || speaker.name || id,
      speeches: counts[id], quota: speaker.quota || (snapshot.ledger && snapshot.ledger[id] || {}).quota });
    view.card.dataset.turnId = turn.turn_id || `${id}:${kind}:${turn.round || 0}:${counts[id]}`;
    view.content.innerHTML = renderMarkdownLite(turn.content || '');
    view.nameTag.textContent = '已发言';
    view.card.classList.remove('thinking');
  }
  for (const speaker of rtState._speakers) {
    const seat = _rtSeat(speaker.id);
    if (!seat) continue;
    const entry = snapshot.ledger && snapshot.ledger[speaker.id];
    const speeches = entry ? Number(entry.speeches) || 0 : counts[speaker.id] || 0;
    const quota = entry && entry.quota || speaker.quota;
    seat.querySelectorAll('.rt-seat-quota i').forEach((dot, index) => dot.classList.toggle('on', index < speeches));
    seat.classList.toggle('spent', speeches >= quota);
  }
  if (rtSummary) { rtSummary.hidden = true; rtSummary.innerHTML = ''; }
  if (snapshot.status === 'completed') {
    _rtStageTeardown();
    _rtBarText('会议已结束。');
    if (snapshot.summary && Object.keys(snapshot.summary).length) _rtRenderSummary(snapshot.summary);
  }
  _rtScroll();
}

// ============================================================
// 围坐舞台：开会期间摆在上面的"圆桌"。谁发言谁起身（带声浪），
// 谁想说话谁举手（冒「想发言」签），额度用完的坐暗。
// 数据全靠 start 事件的 speakers 摆台，之后按事件流换状态。
// ============================================================
function _rtStageBuild(speakers, picker, topic) {
  if (!rtStage) return;
  rtStage.innerHTML = '';
  rtStage.classList.remove('done', 'contention');
  rtState._stage = new Map();

  const seats = document.createElement('div');
  seats.className = 'rt-stage-seats';
  (speakers || []).forEach((sp) => {
    const seat = document.createElement('div');
    seat.className = 'rt-seat theme-' + (sp.theme || 'original');
    seat.dataset.id = sp.id;
    const quota = Math.max(1, (sp.quota | 0) || 1);
    let dots = '';
    for (let i = 0; i < quota; i++) dots += '<i></i>';
    seat.innerHTML = `
      <span class="rt-seat-hand">想发言</span>
      <span class="rt-seat-avatar">${escapeHtml(sp.avatar || nameMark(sp.name))}</span>
      <span class="rt-seat-waves"><i></i><i></i><i></i></span>
      <span class="rt-seat-name">${escapeHtml(sp.name)}</span>
      <span class="rt-seat-quota" title="发言额度">${dots}</span>`;
    seats.appendChild(seat);
    rtState._stage.set(sp.id, seat);
  });
  rtStage.appendChild(seats);

  const table = document.createElement('div');
  table.className = 'rt-stage-table';
  const topicEl = document.createElement('div');
  topicEl.className = 'rt-stage-topic';
  topicEl.textContent = topic || '';
  topicEl.title = topic || '';
  table.appendChild(topicEl);
  rtStage.appendChild(table);

  const host = document.createElement('div');
  host.className = 'rt-stage-host';
  host.textContent = picker === 'agent' ? '主持人代班' : '你主持';
  rtStage.appendChild(host);
  rtState._stageHost = host;

  // 开会期间设置面板让位给舞台；散场时再还回来
  if (rtSetup) rtSetup.hidden = true;
  rtStage.hidden = false;
}

function _rtSeat(id) {
  return rtState._stage && id ? rtState._stage.get(id) : null;
}

function _rtStageEach(fn) {
  if (rtState._stage) rtState._stage.forEach(fn);
}

function _rtStageSetHost(active) {
  if (rtState._stageHost) rtState._stageHost.classList.toggle('active', !!active);
}

// 散场：全体落座、撤掉进行中的高亮；摆台留着当合影，设置面板还回来
function _rtStageTeardown() {
  _rtStageEach(s => s.classList.remove('speaking', 'raised', 'candidate'));
  _rtStageSetHost(false);
  if (rtStage && !rtStage.hidden) rtStage.classList.add('done');
  if (rtSetup) rtSetup.hidden = false;
}

// 开新一场 / 离开视图：连摆台一起撤
function _rtStageReset() {
  rtState._stage = null;
  rtState._stageHost = null;
  if (rtStage) {
    rtStage.hidden = true;
    rtStage.innerHTML = '';
    rtStage.classList.remove('done', 'contention');
  }
  if (rtSetup) rtSetup.hidden = false;
}

// ============================================================
// 点将弹层：收齐请缨后，让用户阅读并选择；等待期间不自动代选。
// ============================================================
function _rtAskShow(request, error = '') {
  return new Promise((resolve) => {
    const candidates = Array.isArray(request.candidates) ? request.candidates : [];
    const allowed = new Map(candidates.map(c => [c.id, c]));
    const receipts = Array.isArray(request.intents) ? request.intents : [];
    const seen = new Set();
    const rows = receipts.map(receipt => {
      const id = receipt.character_id || receipt.id;
      seen.add(id);
      return { ...receipt, ...(allowed.get(id) || {}), id };
    });
    candidates.forEach(c => { if (!seen.has(c.id)) rows.push(c); });
    rtAskList.innerHTML = '';
    rows.forEach(c => {
      const failed = c.status === 'error';
      const selectable = allowed.has(c.id) && !failed;
      const card = document.createElement('article');
      card.className = 'rt-ask-item' + (selectable ? '' : ' rt-ask-unavailable');
      const move = { rebut: '反驳', clarify: '澄清', distinguish: '提出区分', ask: '追问' }[c.move] || c.move || '';
      const details = [];
      if (c.urgency != null) details.push(`紧迫度 ${c.urgency}/5`);
      if (move) details.push(move);
      if (c.quota != null) details.push(`已发言 ${c.speeches || 0}/${c.quota} 次`);
      const target = c.target || {};
      card.innerHTML = `
        <div class="rt-ask-person"><span class="rt-ask-name">${escapeHtml(c.name || c.id || '')}</span>
          <span class="rt-ask-status">${failed ? '请缨失败' : selectable ? '请缨发言' : '暂不发言'}</span></div>
        ${details.length ? `<p class="rt-ask-meta">${escapeHtml(details.join(' · '))}</p>` : ''}
        ${c.reason ? `<p class="rt-ask-reason">${escapeHtml(c.reason)}</p>` : ''}
        ${target.speaker ? `<p class="rt-ask-target">回应对象：${escapeHtml(target.speaker)}</p>` : ''}
        ${target.quote ? `<blockquote class="rt-ask-quote">${escapeHtml(target.quote)}</blockquote>` : ''}
        ${c.error ? `<p class="rt-ask-error-detail">${escapeHtml(c.error)}</p>` : ''}
        ${c.parse_error ? `<p class="rt-ask-error-detail">请缨信息读取提示：${escapeHtml(c.parse_error)}</p>` : ''}`;
      if (selectable) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'rt-ask-select';
        btn.dataset.id = c.id;
        btn.textContent = `请${c.name || c.id}发言`;
        btn.addEventListener('click', () => _rtAskResolve(c.id));
        card.appendChild(btn);
      }
      rtAskList.appendChild(card);
    });
    const title = document.getElementById('rtAskTitle');
    if (title) title.textContent = candidates.length ? '请缨已齐，你想先听谁？' : '本轮无人请缨';
    const hint = document.getElementById('rtAskHint');
    if (hint) hint.textContent = candidates.length ? '阅读后点将，或结束讨论。不会自动选择。' : '可查看各位的回执，再点击结束讨论。';
    const errorEl = document.getElementById('rtAskError');
    if (errorEl) { errorEl.textContent = error; errorEl.hidden = !error; }
    if (rtAskCancel) rtAskCancel.disabled = false;
    rtAsk.hidden = false;
    rtState._askResolve = resolve;
  });
}

function _rtAskResolve(id) {
  if (rtAsk && !rtAsk.hidden) rtAsk.hidden = true;
  const r = rtState._askResolve;
  rtState._askResolve = null;
  if (r) r(id);
}

async function _rtSendChoice(id, decisionId, runId = rtState._runId) {
  // null 仅用于关闭面板/离开页面，不是放弃点将，也不是结束讨论。
  if (!id || !rtState.sessionId || runId !== rtState._runId) return;
  if (!decisionId) throw new Error('当前点将信息已失效，请刷新会议');
  const resp = await fetch('/persona/roundtable/' + encodeURIComponent(rtState.sessionId) + '/choice', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ character_id: id, decision_id: decisionId }),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `点将未成功（HTTP ${resp.status}）`);
  }
  await _rtConsumeStream(resp, runId);
}

async function _rtSnapshot(sessionId) {
  const resp = await fetch('/persona/roundtable/' + encodeURIComponent(sessionId) + '/snapshot');
  if (!resp.ok) {
    const error = new Error(`暂时无法读取会议（HTTP ${resp.status}）`);
    error.status = resp.status;
    throw error;
  }
  return await resp.json();
}

function _rtRecoveryResolve(action) {
  const panel = document.getElementById('rtRecovery');
  if (panel) panel.hidden = true;
  const resolve = rtState._recoveryResolve;
  rtState._recoveryResolve = null;
  if (resolve) resolve(action);
}

function _rtRecoveryShow(status, message = '') {
  return new Promise(resolve => {
    const panel = document.getElementById('rtRecovery');
    const text = document.getElementById('rtRecoveryText');
    const continueButton = document.getElementById('rtContinueBtn');
    const statusText = status === 'running'
      ? '会议正在生成。刷新进度后会补齐发言，不会重复提交点将。'
      : status === 'interrupted' ? '上次生成已中断，可以从已保存的进度继续。' : '连接暂时不可用，可以刷新重试。';
    if (text) text.textContent = message ? `${message} ${statusText}` : statusText;
    if (continueButton) continueButton.hidden = status !== 'interrupted';
    if (panel) panel.hidden = false;
    rtState._recoveryResolve = resolve;
  });
}

async function _rtContinueSession(runId) {
  const resp = await fetch('/persona/roundtable/' + encodeURIComponent(rtState.sessionId) + '/continue', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}),
  });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `暂时无法继续生成（HTTP ${resp.status}）`);
  }
  await _rtConsumeStream(resp, runId);
}

async function _rtRecoverSession(sessionId, runId, error = '') {
  while (runId === rtState._runId) {
    let snapshot;
    try {
      snapshot = await _rtSnapshot(sessionId);
    } catch (err) {
      if (runId !== rtState._runId) return;
      if (err.status === 404) {
        _rtNotice('这场会议记录已不存在，可以开始新会议。', 'error');
        _rtStageTeardown();
        try {
          if ((_rtSavedSession() || {}).session_id === sessionId) localStorage.removeItem(RT_SAVED_SESSION_KEY);
        } catch (e) { /* ignore */ }
        _rtUpdateResumeEntry();
        return;
      }
      const action = await _rtRecoveryShow('error', `连接中断：${err.message || String(err)}。请刷新重试。`);
      if (!action || runId !== rtState._runId) return;
      continue;
    }
    if (runId !== rtState._runId) return;
    // 正文、名单和配额都以同一份权威快照为准，再让用户作下一次决定。
    _rtApplySnapshot(snapshot);
    if (snapshot.status === 'completed') return;
    if (snapshot.status === 'waiting' && snapshot.choice_request && snapshot.choice_request.decision_id) {
      await _rtHandleChoice(snapshot.choice_request, runId, error);
      return;
    }
    const status = snapshot.status === 'interrupted' ? 'interrupted' : snapshot.status === 'running' ? 'running' : 'error';
    const action = await _rtRecoveryShow(status, error || (status === 'error' ? '当前点将信息尚未就绪，请刷新会议进度。' : ''));
    if (!action || runId !== rtState._runId) return;
    error = '';
    if (action === 'continue' && status === 'interrupted') {
      try {
        await _rtContinueSession(runId);
        // continue 可能只返回快照（例如另一个 worker 已在运行）。
        // 流结束后再读当前状态，重新进入等待、生成中或完成视图。
        continue;
      } catch (err) {
        if (runId !== rtState._runId) return;
        error = `未能接续讨论：${err.message || String(err)}`;
      }
    }
  }
}

async function resumeRoundtable() {
  if (rtState.loading) return;
  const saved = _rtSavedSession();
  if (!saved) return;
  const runId = ++rtState._runId;
  rtState.loading = true;
  rtStartBtn.disabled = true;
  rtTopic.disabled = true;
  rtState.sessionId = saved.session_id;
  _rtAskResolve(null);
  _rtRecoveryResolve(null);
  _rtBarText('正在读取已保存的会议…');
  try {
    await _rtRecoverSession(saved.session_id, runId);
  } finally {
    if (runId === rtState._runId) {
      rtState.loading = false;
      rtTopic.disabled = false;
      updateRtStartBtn();
      _rtFinalizeCurrent();
    }
  }
}

async function _rtHandleChoice(request, runId, error = '') {
  const sessionId = rtState.sessionId;
  if (!request.decision_id) {
    await _rtRecoverSession(sessionId, runId, '当前点将信息已失效，已重新读取会议。');
    return;
  }
  const chosen = await _rtAskShow(request, error);
  if (!chosen || runId !== rtState._runId) return;
  try {
    await _rtSendChoice(chosen, request.decision_id, runId);
  } catch (err) {
    if (runId !== rtState._runId) return;
    // 包括 409：不得沿用旧轮的令牌，也不得自动重投原选择。
    await _rtRecoverSession(sessionId, runId, `未能接续讨论：${err.message || String(err)}`);
  }
}

// 消费一条 SSE 流：逐条交给 handleRtEvent。
// 「开始一场」和「点将恢复」返回的是同一种流（后者由 POST /choice 触发），
// 所以解析必须共用——否则用户点完将，恢复后的后续事件会全部丢掉。
async function _rtConsumeStream(resp, runId = rtState._runId) {
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let reachedBoundary = false;
  const dispatch = async line => {
    if (!line.startsWith('data: ')) return;
    let data;
    try { data = JSON.parse(line.slice(6)); } catch (e) { return; }
    if (data.type === 'end' || data.type === 'choice_request' || (data.type === 'snapshot' && data.data)) reachedBoundary = true;
    await handleRtEvent(data, runId);
  };
  while (true) {
    const { done, value } = await reader.read();
    if (runId !== rtState._runId) { await reader.cancel(); return; }
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const line of parts) {
      // 点将需要暂停消费、等用户点完再继续；因此这里必须 await
      await dispatch(line);
      if (runId !== rtState._runId) { await reader.cancel(); return; }
    }
  }
  if (buffer.trim().startsWith('data: ')) {
    await dispatch(buffer.trim());
  }
  if (runId === rtState._runId && !reachedBoundary) throw new Error('连接在本轮结束前中断');
}

// ============================================================
// 主流程
// ============================================================
async function startRoundtable() {
  if (rtState.loading) return;
  const topic = rtTopic.value.trim();
  const ids = Array.from(rtState.selected);
  if (ids.length < 2) { toast('请至少选择 2 位与会者'); return; }
  if (ids.length > RT_MAX) { toast(`最多只能选择 ${RT_MAX} 位与会者`); return; }
  if (!topic) { toast('请填写议题'); return; }

  const previousSession = _rtSavedSession();
  const sessionId = crypto.randomUUID();
  const runId = ++rtState._runId;
  rtState.loading = true;
  rtStartBtn.disabled = true;
  rtTopic.disabled = true;
  rtTranscript.innerHTML = '';
  rtState._current = null;
  rtState._bar = null;
  rtState._summaryEl = null;
  rtState.sessionId = sessionId;
  rtState._speakers = [];
  if (rtSummary) { rtSummary.hidden = true; rtSummary.innerHTML = ''; }
  _rtAskResolve(null);
  _rtRecoveryResolve(null);
  _rtStageReset();
  // 在请求发出前保留入口：即使 start 回执丢失，也知道应恢复哪场会议。
  _rtRememberSession(sessionId, topic, 'running');

  const body = {
    session_id: sessionId,
    topic,
    character_ids: ids,
    rounds: rtState.rounds,
    picker: rtState.picker,
  };
  try {
    const resp = await fetch('/persona/roundtable', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      const error = new Error(d.detail || ('HTTP ' + resp.status));
      // 超时、冲突和服务端错误可能发生在已接受之后，仍需查询快照。
      error.creationRejected = [400, 401, 403, 404, 405, 413, 415, 422, 429].includes(resp.status);
      throw error;
    }
    await _rtConsumeStream(resp, runId);
  } catch (err) {
    if (runId !== rtState._runId) return;
    if (err.creationRejected) {
      // 只撤销本次未创建的入口；保留上一场，也不覆盖其他页面新保存的会议。
      if ((_rtSavedSession() || {}).session_id === sessionId) {
        if (previousSession) _rtRememberSession(previousSession.session_id, previousSession.topic, previousSession.status);
        else {
          try { localStorage.removeItem(RT_SAVED_SESSION_KEY); } catch (e) { /* ignore */ }
          _rtUpdateResumeEntry();
        }
      }
      rtState.sessionId = null;
    } else if (rtState.sessionId) {
      await _rtRecoverSession(rtState.sessionId, runId, `连接中断：${err.message || String(err)}`);
      return;
    }
    _rtFinalizeCurrent();
    _rtStageTeardown();
    const div = document.createElement('div');
    div.className = 'rt-speaker';
    div.innerHTML = `<div class="rt-speaker-body"><div class="rt-speaker-name">出错了</div>
      <div class="rt-speaker-text">${escapeHtml(err.message || String(err))}</div></div>`;
    rtTranscript.appendChild(div);
    _rtScroll();
  } finally {
    if (runId === rtState._runId) {
      _rtAskResolve(null);
      rtState.loading = false;
      rtTopic.disabled = false;
      updateRtStartBtn();
      _rtFinalizeCurrent();
    }
  }
}

async function handleRtEvent(data, runId = rtState._runId) {
  if (runId !== rtState._runId) return;
  switch (data.type) {
    case 'start':
      rtState.sessionId = data.session_id || null;
      rtState._speakers = Array.isArray(data.speakers) ? data.speakers : [];
      _rtRememberSession(rtState.sessionId, data.topic, 'running');
      _rtStageBuild(data.speakers, data.picker, data.topic);
      break;

    case 'snapshot':
      if (data.data) _rtApplySnapshot(data.data);
      break;

    case 'round':
      _rtAppendDivider(data.phase, data.round, data.total);
      break;

    case 'bidding': {
      _rtBarText(`第 ${data.round} 轮：各位在斟酌，此刻有没有非说不可的话…`);
      _rtStageEach(s => s.classList.remove('raised', 'candidate'));
      if (rtStage) rtStage.classList.remove('contention');
      _rtStageSetHost(false);
      // 不在候选名单里的 = 上轮刚说完或额度已用完，本轮不参与投标，坐暗
      const bidders = new Set((data.candidates || []).map(c => c.id));
      _rtStageEach((s, id) => s.classList.toggle('spent', !bidders.has(id)));
      break;
    }

    case 'intent': {
      // 只展示"想说话"的人：愿不愿意开口本身就是看点，沉默的人不占版面
      if (data.speak || data.want) _rtBarChip(data.name, data.reason);
      const seat = _rtSeat(data.character_id);
      if (seat) seat.classList.toggle('raised', !!(data.want || data.speak));
      break;
    }

    case 'choice_request': {
      _rtRememberSession(rtState.sessionId, rtTopic.value, 'waiting');
      _rtBarText(`第 ${data.round} 轮：请缨已收齐，等你点将或结束讨论。`);
      if (rtStage) rtStage.classList.add('contention');
      (data.candidates || []).forEach((c) => {
        const seat = _rtSeat(c.id);
        if (seat) seat.classList.add('candidate');
      });
      _rtStageSetHost(true);
      await _rtHandleChoice(data, runId);
      break;
    }

    case 'choice': {
      const name = _rtNameOf(data.character_id);
      const by = data.by === 'user' ? '你指定'
        : data.by === 'bid' ? '请缨'
          : (data.by === 'auto' ? '自行举手' : '主持人代定');
      _rtBarText(`本轮由 ${name} 发言（${by}${data.note ? ' · ' + data.note : ''}）`);
      _rtStageEach(s => s.classList.remove('raised', 'candidate'));
      if (rtStage) rtStage.classList.remove('contention');
      _rtStageSetHost(false);
      break;
    }

    case 'converged':
      _rtNotice(data.reason === 'quota'
        ? '发言额度已用完，讨论结束。'
        : data.reason === 'user'
          ? '你已结束本场讨论。'
        : data.reason === 'rounds'
          ? '本场交锋轮数已完成，讨论结束。'
          : data.reason === 'budget'
            ? '本场调用已达上限，讨论结束。'
            : '暂时没有新的论点，讨论到此结束。', 'calm');
      _rtStageEach(s => s.classList.remove('raised', 'candidate', 'speaking'));
      break;

    case 'budget_exhausted':
      _rtNotice('本场调用已达上限，会议就此收束。', 'calm');
      _rtStageEach(s => s.classList.remove('raised', 'candidate', 'speaking'));
      break;

    case 'speaker_start': {
      _rtFinalizeCurrent();
      // 舞台：上一位落座，这一位起身、额度点亮一颗
      _rtStageEach(s => s.classList.remove('speaking'));
      const seat = _rtSeat(data.character_id);
      if (seat) {
        seat.classList.add('speaking');
        seat.classList.remove('raised', 'candidate', 'spent');
        const dots = seat.querySelectorAll('.rt-seat-quota i');
        dots.forEach((d, i) => d.classList.toggle('on', i < (data.speeches || 0)));
      }
      rtState._current = _rtCreateSpeechCard(data);
      _rtScroll();
      break;
    }

    case 'token': {
      const cur = rtState._current;
      if (!cur) break;
      cur.text += (data.content || '');
      cur.content.innerHTML = renderMarkdownLite(cur.text);
      _rtScroll();
      break;
    }

    case 'speaker_end': {
      const seat = _rtSeat(data.character_id);
      if (seat) seat.classList.remove('speaking');
      const cur = rtState._current;
      if (cur) {
        cur.text = data.content || cur.text;
        cur.content.innerHTML = renderMarkdownLite(cur.text);
        if (cur.nameTag) cur.nameTag.textContent = '已发言';
        cur.card.classList.remove('thinking');
        rtState._current = null;
      }
      _rtScroll();
      break;
    }

    case 'end':
      if (Array.isArray(data.transcript)) _rtApplySnapshot({ ...data, status: 'completed' });
      _rtRememberSession(rtState.sessionId, data.topic || rtTopic.value, 'completed');
      _rtBarText('会议结束。');
      _rtStageTeardown();
      _rtScroll();
      break;

    case 'summary_start':
      // 只有服务端实际开始生成纪要才显示等待；完成快照可能没有纪要。
      if (rtSummary) {
        rtSummary.hidden = false;
        rtSummary.innerHTML = '<div class="rt-summary-loading">主持人正在整理各方观点…</div>';
      }
      _rtScroll();
      break;

    case 'summary':
      _rtRenderSummary(data.data || {});
      break;

    case 'summary_error':
      if (rtSummary) {
        rtSummary.hidden = false;
        rtSummary.innerHTML = `<div class="rt-summary-loading">${escapeHtml(data.content || '主持人总结暂时无法生成。')}</div>`;
      }
      break;

    case 'summary_end':
      break;

    case 'error': {
      _rtFinalizeCurrent();
      _rtStageTeardown();
      const div = document.createElement('div');
      div.className = 'rt-speaker';
      div.innerHTML = `<div class="rt-speaker-body"><div class="rt-speaker-name">出错了</div>
        <div class="rt-speaker-text">${escapeHtml(data.content || '未知错误')}</div></div>`;
      rtTranscript.appendChild(div);
      _rtScroll();
      break;
    }

    default:
      break;
  }
}

function _rtNameOf(id) {
  const c = (state.characters || []).find(x => x.id === id) || rtState._speakers.find(x => x.id === id);
  return c ? c.name : id;
}

// ============================================================
// 主持人总结
//
// 保留各方立场与分歧，按发言中明确的条件说明如何运用，不以一致为结束前提。
// ============================================================
function _rtRenderSummary(d) {
  if (!rtSummary) return;
  const clashes = Array.isArray(d.clashes) ? d.clashes : [];
  const positions = Array.isArray(d.positions) ? d.positions : [];
  const openItems = Array.isArray(d.open) ? d.open : [];
  const applications = Array.isArray(d.applications) ? d.applications : [];
  const closing = typeof d.closing === 'string' ? d.closing.trim() : '';

  let html = '<div class="rt-summary-card">';
  html += `<div class="rt-summary-head">
      <span class="rt-summary-title">主持人总结</span>
      <span class="rt-summary-note">保留分歧，不判胜负</span>
    </div>`;
  if (d.topic) {
    html += `<p class="rt-summary-topic">议题：${escapeHtml(d.topic)}</p>`;
  }
  if (closing) {
    html += `<p class="rt-summary-closing">${escapeHtml(closing)}</p>`;
  }

  if (clashes.length) {
    html += '<p class="rt-summary-label">交锋线</p><div class="rt-summary-clashes">';
    clashes.forEach((c) => {
      html += `<div class="rt-clash">
          <span class="rt-clash-a">${escapeHtml(c.a)}</span>
          <span class="rt-clash-vs">对</span>
          <span class="rt-clash-b">${escapeHtml(c.b)}</span>
          <span class="rt-clash-point">${escapeHtml(c.point)}</span>
        </div>`;
    });
    html += '</div>';
  }

  if (positions.length) {
    html += '<p class="rt-summary-label">各方主张</p><div class="rt-summary-positions">';
    positions.forEach((p) => {
      html += `<div class="rt-position">
          <span class="rt-position-name">${escapeHtml(p.name)}</span>
          <span class="rt-position-stance">${escapeHtml(p.stance)}</span>
        </div>`;
    });
    html += '</div>';
  }

  if (applications.length) {
    html += '<p class="rt-summary-label">观点的适用条件</p><div class="rt-summary-applications">';
    applications.forEach((a) => {
      html += `<div class="rt-application">
          <span class="rt-position-name">${escapeHtml(a.name)}</span>
          <div class="rt-application-body">
            <span class="rt-application-scenario">${escapeHtml(a.scenario)}</span>
            <span class="rt-position-stance">${escapeHtml(a.application)}</span>
          </div>
        </div>`;
    });
    html += '</div>';
  }

  if (openItems.length) {
    html += '<p class="rt-summary-label">悬而未决</p><div class="rt-summary-open">';
    openItems.forEach((q, i) => {
      html += `<div class="rt-open-item"><span class="rt-open-idx">${i + 1}</span><span>${escapeHtml(q)}</span></div>`;
    });
    html += '</div>';
  }

  if (d.degraded) {
    html += '<p class="rt-summary-degraded">本次总结仅保留发言记录中的主张，适用条件暂未整理。</p>';
  }

  // 发言次数：这是**事实**不是分数（配额本来就是公开机制）。
  // 摆出来是为了避免误读——某人说得少，可能只是额度用完了，不是无话可说。
  const counts = d.speeches || [];
  if (counts.length) {
    html += '<p class="rt-summary-counts">本场发言次数：'
      + counts.map(s => `${escapeHtml(s.name)} ${s.speeches || 0}/${s.quota || 0}`).join(' · ')
      + '</p>';
  }

  // 自表态：系统不下判断，但给你一个出口。纯前端，不上报、不显示全局占比。
  if (positions.length) {
    html += '<div class="rt-stand">';
    html += '<p class="rt-summary-label">你自己怎么看</p>';
    html += '<div class="rt-stand-list">';
    // 用下标而不是把名字塞进 data-* —— escapeHtml 走 textContent，不转义引号，
    // 自建角色名里带引号会撑破属性（XSS）。名字在点击时按下标回查。
    positions.forEach((p, i) => {
      html += `<button class="rt-stand-btn" type="button" data-idx="${i}">${escapeHtml(p.name)}</button>`;
    });
    html += '<button class="rt-stand-btn rt-stand-nobody" type="button" data-idx="-1">按情境选择</button>';
    html += '</div>';
    html += '<p class="rt-stand-note" id="rtStandNote">可以保留分歧，也可以按情境取舍。</p>';
    html += '</div>';
  }

  html += '</div>';
  rtSummary.hidden = false;
  rtSummary.innerHTML = html;

  rtSummary.querySelectorAll('.rt-stand-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      rtSummary.querySelectorAll('.rt-stand-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const note = rtSummary.querySelector('#rtStandNote');
      const idx = parseInt(btn.dataset.idx, 10);
      const picked = idx >= 0 ? (positions[idx] || {}).name : '';
      if (note) {
        note.textContent = picked
          ? `你选了 ${picked} 这一边。这个判断只在你这里。`
          : '你会结合具体情境选择观点。这个判断只在你这里。';
      }
    });
  });
  rtSummary.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// 事件绑定
// ============================================================
// 「争鸣」没有独立入口按钮：入口是首页分区层的第三张卡
// （.zone-card[data-mode=roundtable]），由 home.js 直接调 openRoundtable()。
function leaveRoundtable() {
  closeRoundtable();
  // 争鸣是从首页分区层直接进入的，返回就回到分区选择
  if (homeView.classList.contains('hidden')) {
    showHome();
  } else {
    homeStage('zone');
  }
}
if (rtBackBtn) rtBackBtn.addEventListener('click', leaveRoundtable);
const rtAskLeaveBtn = document.getElementById('rtAskLeaveBtn');
if (rtAskLeaveBtn) rtAskLeaveBtn.addEventListener('click', leaveRoundtable);
if (rtStartBtn) rtStartBtn.addEventListener('click', startRoundtable);
if (rtTopic) rtTopic.addEventListener('input', updateRtStartBtn);
if (rtAskCancel) rtAskCancel.addEventListener('click', () => _rtAskResolve('__end__'));
const rtResumeBtn = document.getElementById('rtResumeBtn');
if (rtResumeBtn) rtResumeBtn.addEventListener('click', resumeRoundtable);
const rtRefreshBtn = document.getElementById('rtRefreshBtn');
if (rtRefreshBtn) rtRefreshBtn.addEventListener('click', () => _rtRecoveryResolve('refresh'));
const rtContinueBtn = document.getElementById('rtContinueBtn');
if (rtContinueBtn) rtContinueBtn.addEventListener('click', () => _rtRecoveryResolve('continue'));
if (rtRounds) rtRounds.addEventListener('click', (e) => {
  const btn = e.target.closest('.rt-round-btn');
  if (!btn) return;
  rtState.rounds = parseInt(btn.dataset.rounds, 10) || 3;
  rtRounds.querySelectorAll('.rt-round-btn').forEach(b => b.classList.toggle('active', b === btn));
});
if (rtPicker) rtPicker.addEventListener('click', (e) => {
  const btn = e.target.closest('.rt-round-btn');
  if (!btn) return;
  rtState.picker = btn.dataset.picker === 'agent' ? 'agent' : 'user';
  rtPicker.querySelectorAll('.rt-round-btn').forEach(b => b.classList.toggle('active', b === btn));
});
