// ============================================================
// 圆桌会议（Roundtable）
// ============================================================
const rtState = {
  selected: new Set(),   // 已选与会者 id
  rounds: 1,             // 交锋轮数
  loading: false,
  _current: null,        // 当前正在流式渲染的发言卡 { card, content }
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

function openRoundtable() {
  closeSidebar();
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
  roundtableView.hidden = true;
}

function _rtRoundLabel(phase, round, total) {
  if (phase === 'opening') return '开场陈述';
  return `第 ${round}/${total} 轮 · 交锋`;
}

function _rtAppendDivider(phase, round, total) {
  const div = document.createElement('div');
  div.className = 'rt-round-divider';
  div.textContent = _rtRoundLabel(phase, round, total);
  rtTranscript.appendChild(div);
  rtTranscript.scrollTop = rtTranscript.scrollHeight;
}

function _rtFinalizeCurrent() {
  if (rtState._current) {
    rtState._current.card.classList.remove('thinking');
    rtState._current = null;
  }
}

async function startRoundtable() {
  if (rtState.loading) return;
  const topic = rtTopic.value.trim();
  const ids = Array.from(rtState.selected);
  if (ids.length < 2) { toast('请至少选择 2 位与会者'); return; }
  if (ids.length > RT_MAX) { toast(`最多只能选择 ${RT_MAX} 位与会者`); return; }
  if (!topic) { toast('请填写辩论议题'); return; }

  rtState.loading = true;
  rtStartBtn.disabled = true;
  rtTopic.disabled = true;
  rtTranscript.innerHTML = '';
  rtState._current = null;

  const body = { topic, character_ids: ids, rounds: rtState.rounds };
  try {
    const resp = await fetch('/persona/roundtable', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
        handleRtEvent(data);
      }
    }
    if (buffer.trim().startsWith('data: ')) {
      try {
        const data = JSON.parse(buffer.trim().slice(6));
        handleRtEvent(data);
      } catch (e) { /* ignore */ }
    }
  } catch (err) {
    _rtFinalizeCurrent();
    const div = document.createElement('div');
    div.className = 'rt-speaker';
    div.innerHTML = `<div class="rt-speaker-body"><div class="rt-speaker-name">出错了</div>
      <div class="rt-speaker-text">${escapeHtml(err.message || String(err))}</div></div>`;
    rtTranscript.appendChild(div);
    rtTranscript.scrollTop = rtTranscript.scrollHeight;
  } finally {
    rtState.loading = false;
    rtStartBtn.disabled = false;
    rtTopic.disabled = false;
    _rtFinalizeCurrent();
  }
}

function handleRtEvent(data) {
  switch (data.type) {
    case 'round':
      _rtAppendDivider(data.phase, data.round, data.total);
      break;
    case 'speaker_start': {
      _rtFinalizeCurrent();
      const card = document.createElement('div');
      card.className = 'rt-speaker theme-' + (data.theme || 'original') + ' thinking';
      card.innerHTML = `
        <div class="rt-speaker-avatar">${escapeHtml(nameMark(data.name))}</div>
        <div class="rt-speaker-body">
          <div class="rt-speaker-name">${escapeHtml(data.name)} <span class="ai-tag" title="本条内容由人工智能生成（AI-generated content）">AI 生成</span> <span class="rt-speaker-tag">正在发言…</span></div>
          <div class="rt-speaker-text"></div>
        </div>`;
      rtTranscript.appendChild(card);
      rtTranscript.scrollTop = rtTranscript.scrollHeight;
      rtState._current = {
        card,
        content: card.querySelector('.rt-speaker-text'),
        nameTag: card.querySelector('.rt-speaker-tag'),
        text: '',
      };
      break;
    }
    case 'token': {
      const cur = rtState._current;
      if (!cur) break;
      cur.text += (data.content || '');
      cur.content.innerHTML = renderMarkdownLite(cur.text);
      rtTranscript.scrollTop = rtTranscript.scrollHeight;
      break;
    }
    case 'speaker_end': {
      const cur = rtState._current;
      if (cur) {
        cur.text = data.content || cur.text;
        cur.content.innerHTML = renderMarkdownLite(cur.text);
        if (cur.nameTag) cur.nameTag.textContent = '已发言';
        cur.card.classList.remove('thinking');
        rtState._current = null;
      }
      rtTranscript.scrollTop = rtTranscript.scrollHeight;
      break;
    }
    case 'error': {
      _rtFinalizeCurrent();
      const div = document.createElement('div');
      div.className = 'rt-speaker';
      div.innerHTML = `<div class="rt-speaker-body"><div class="rt-speaker-name">出错了</div>
        <div class="rt-speaker-text">${escapeHtml(data.content || '未知错误')}</div></div>`;
      rtTranscript.appendChild(div);
      rtTranscript.scrollTop = rtTranscript.scrollHeight;
      break;
    }
    case 'end':
      rtTranscript.scrollTop = rtTranscript.scrollHeight;
      break;
    default:
      break;
  }
}

if (roundtableBtn) roundtableBtn.addEventListener('click', openRoundtable);
if (introRoundtableBtn) introRoundtableBtn.addEventListener('click', openRoundtable);
if (rtBackBtn) rtBackBtn.addEventListener('click', () => {
  closeRoundtable();
  // 若在首页内进入圆桌（homeView 可见），返回到对应区域的角色列表；否则回到首页区域选择
  if (homeView.classList.contains('hidden')) {
    showHome();
  } else {
    homeStage('select');
    renderDeck();
  }
});
if (rtStartBtn) rtStartBtn.addEventListener('click', startRoundtable);
if (rtTopic) rtTopic.addEventListener('input', updateRtStartBtn);
if (rtRounds) rtRounds.addEventListener('click', (e) => {
  const btn = e.target.closest('.rt-round-btn');
  if (!btn) return;
  rtState.rounds = parseInt(btn.dataset.rounds, 10) || 1;
  rtRounds.querySelectorAll('.rt-round-btn').forEach(b => b.classList.toggle('active', b === btn));
});
