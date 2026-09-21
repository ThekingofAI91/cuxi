// ============================================================
// 争鸣（圆桌会议）
//
// 与「问道 / 会心」同级的一种玩法：请两三位智者同桌，就一个议题各自判断要不要
// 开口。多人同时举手时由用户点将；用户不想管，就让主持人代班。
// 结尾只有纪要（分歧地图），不判胜负。
// ============================================================
const rtState = {
  selected: new Set(),   // 已选与会者 id
  rounds: 1,             // 交锋轮数
  picker: 'user',        // 谁主持：user = 用户点将 / agent = 主持人代班
  loading: false,
  sessionId: null,       // 本场会议 id（点将端点需要它）
  _current: null,        // 当前正在流式渲染的发言卡 { card, content }
  _bar: null,            // 本轮公告条（谁想说话 / 最后谁发言）
  _askTimer: null,
  _askResolve: null,
  _summaryEl: null,
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
  // 每次进来都是一场新会议：清掉上一场的发言与纪要
  rtState.sessionId = null;
  rtState._current = null;
  rtState._bar = null;
  rtState._summaryEl = null;
  _rtAskResolve(null);
  if (rtTranscript) rtTranscript.innerHTML = '';
  if (rtSummary) { rtSummary.hidden = true; rtSummary.innerHTML = ''; }

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
  _rtAskResolve(null);
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

// ============================================================
// 点将弹层：多人同时举手时，让用户选谁先说
// ============================================================
function _rtAskShow(candidates, timeout, defaultId) {
  return new Promise((resolve) => {
    rtAskList.innerHTML = '';
    candidates.forEach((c) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'rt-ask-item' + (c.id === defaultId ? ' suggested' : '');
      btn.dataset.id = c.id;
      btn.innerHTML = `
        <span class="rt-ask-name">${escapeHtml(c.name)}</span>
        <span class="rt-ask-reason">${escapeHtml(c.reason || '有话要说')}</span>`;
      btn.addEventListener('click', () => _rtAskResolve(c.id));
      rtAskList.appendChild(btn);
    });
    rtAsk.hidden = false;
    let left = Math.max(1, timeout | 0);
    rtAskTimer.textContent = left + ' 秒';
    clearInterval(rtState._askTimer);
    rtState._askTimer = setInterval(() => {
      left -= 1;
      if (left <= 0) { _rtAskResolve(null); return; }
      rtAskTimer.textContent = left + ' 秒';
    }, 1000);
    rtState._askResolve = resolve;
  });
}

function _rtAskResolve(id) {
  if (rtState._askTimer) { clearInterval(rtState._askTimer); rtState._askTimer = null; }
  if (rtAsk && !rtAsk.hidden) rtAsk.hidden = true;
  const r = rtState._askResolve;
  rtState._askResolve = null;
  if (r) r(id);
}

async function _rtSendChoice(id) {
  if (!rtState.sessionId) return;
  try {
    await fetch('/persona/roundtable/' + encodeURIComponent(rtState.sessionId) + '/choice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // 空字符串 = 主动弃权，交给主持人代班（后端按"无人选择"处理）
      body: JSON.stringify({ character_id: id || '' }),
    });
  } catch (err) {
    // 会议可能已经结束（或已被主持人接管），点将失败不影响观感，忽略
  }
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

  rtState.loading = true;
  rtStartBtn.disabled = true;
  rtTopic.disabled = true;
  rtTranscript.innerHTML = '';
  rtState._current = null;
  rtState._bar = null;
  rtState._summaryEl = null;
  if (rtSummary) { rtSummary.hidden = true; rtSummary.innerHTML = ''; }
  _rtAskResolve(null);

  const body = {
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
        // 点将需要暂停消费、等用户点完再继续；因此这里必须 await
        await handleRtEvent(data);
      }
    }
    if (buffer.trim().startsWith('data: ')) {
      try {
        const data = JSON.parse(buffer.trim().slice(6));
        await handleRtEvent(data);
      } catch (e) { /* ignore */ }
    }
  } catch (err) {
    _rtFinalizeCurrent();
    const div = document.createElement('div');
    div.className = 'rt-speaker';
    div.innerHTML = `<div class="rt-speaker-body"><div class="rt-speaker-name">出错了</div>
      <div class="rt-speaker-text">${escapeHtml(err.message || String(err))}</div></div>`;
    rtTranscript.appendChild(div);
    _rtScroll();
  } finally {
    _rtAskResolve(null);
    rtState.loading = false;
    rtStartBtn.disabled = false;
    rtTopic.disabled = false;
    _rtFinalizeCurrent();
  }
}

async function handleRtEvent(data) {
  switch (data.type) {
    case 'start':
      rtState.sessionId = data.session_id || null;
      break;

    case 'round':
      _rtAppendDivider(data.phase, data.round, data.total);
      break;

    case 'intent_start':
      _rtBarText('各位正在斟酌，此刻有没有非说不可的话…');
      break;

    case 'intent':
      // 只展示"想说话"的人：愿不愿意开口本身就是看点，沉默的人不占版面
      if (data.speak) _rtBarChip(data.name, data.reason);
      break;

    case 'contention': {
      _rtBarText('多人同时要发言——你来点将。');
      const chosen = await _rtAskShow(
        data.candidates || [], data.timeout || 25, data.moderator_default);
      await _rtSendChoice(chosen);
      break;
    }

    case 'choice': {
      const name = _rtNameOf(data.character_id);
      const by = data.by === 'user' ? '你指定' : (data.by === 'auto' ? '自行举手' : '主持人代定');
      _rtBarText(`本轮由 ${name} 发言（${by}${data.note ? ' · ' + data.note : ''}）`);
      break;
    }

    case 'converged':
      _rtNotice(data.reason === 'quota'
        ? '发言额度已用完，会议到此为止。'
        : '再无人举手，讨论自然收敛于此。', 'calm');
      break;

    case 'budget_exhausted':
      _rtNotice('本场调用已达上限，会议就此收束。', 'calm');
      break;

    case 'speaker_start': {
      _rtFinalizeCurrent();
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
      _rtScroll();
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
      _rtScroll();
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
      _rtScroll();
      break;
    }

    case 'end':
      _rtBarText('会议结束。');
      // 纪要在 end 之后异步补推（非阻塞旁路），这里先立占位
      if (rtSummary) {
        rtSummary.hidden = false;
        rtSummary.innerHTML = '<div class="rt-summary-loading">正在整理纪要…</div>';
      }
      _rtScroll();
      break;

    case 'summary':
      _rtRenderSummary(data.data || {});
      break;

    case 'summary_error':
      if (rtSummary) {
        rtSummary.hidden = false;
        rtSummary.innerHTML = `<div class="rt-summary-loading">${escapeHtml(data.content || '纪要暂时无法生成。')}</div>`;
      }
      break;

    case 'summary_end':
      break;

    case 'error': {
      _rtFinalizeCurrent();
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
  const c = (state.characters || []).find(x => x.id === id);
  return c ? c.name : id;
}

// ============================================================
// 纪要（分歧地图）
//
// 系统不判胜负，只把"你们在哪吵、吵的是什么"摆清楚；怎么想留给你。
// 因此这里没有胜负位，只有：交锋线 / 各方主张 / 悬而未决 / 你自己怎么看。
// ============================================================
function _rtRenderSummary(d) {
  if (!rtSummary) return;
  const clashes = d.clashes || [];
  const positions = d.positions || [];
  const openItems = d.open || [];

  let html = '<div class="rt-summary-card">';
  html += `<div class="rt-summary-head">
      <span class="rt-summary-title">纪要</span>
      <span class="rt-summary-note">只记录，不判胜负</span>
    </div>`;
  if (d.topic) {
    html += `<p class="rt-summary-topic">议题：${escapeHtml(d.topic)}</p>`;
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

  if (openItems.length) {
    html += '<p class="rt-summary-label">悬而未决</p><div class="rt-summary-open">';
    openItems.forEach((q, i) => {
      html += `<div class="rt-open-item"><span class="rt-open-idx">${i + 1}</span><span>${escapeHtml(q)}</span></div>`;
    });
    html += '</div>';
  }

  if (d.degraded) {
    html += '<p class="rt-summary-degraded">本次纪要只保留了各方主张。</p>';
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
    html += '<button class="rt-stand-btn rt-stand-nobody" type="button" data-idx="-1">都有一点</button>';
    html += '</div>';
    html += '<p class="rt-stand-note" id="rtStandNote">不必有结论——想清楚就好。</p>';
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
          : '你觉得两边都有道理。这个判断只在你这里。';
      }
    });
  });
  rtSummary.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// 事件绑定
// ============================================================
if (roundtableBtn) roundtableBtn.addEventListener('click', openRoundtable);
if (introRoundtableBtn) introRoundtableBtn.addEventListener('click', openRoundtable);
if (rtBackBtn) rtBackBtn.addEventListener('click', () => {
  closeRoundtable();
  // 争鸣是从首页分区层直接进入的，返回就回到分区选择
  if (homeView.classList.contains('hidden')) {
    showHome();
  } else {
    homeStage('zone');
  }
});
if (rtStartBtn) rtStartBtn.addEventListener('click', startRoundtable);
if (rtTopic) rtTopic.addEventListener('input', updateRtStartBtn);
if (rtAskCancel) rtAskCancel.addEventListener('click', () => _rtAskResolve(null));
if (rtRounds) rtRounds.addEventListener('click', (e) => {
  const btn = e.target.closest('.rt-round-btn');
  if (!btn) return;
  rtState.rounds = parseInt(btn.dataset.rounds, 10) || 1;
  rtRounds.querySelectorAll('.rt-round-btn').forEach(b => b.classList.toggle('active', b === btn));
});
if (rtPicker) rtPicker.addEventListener('click', (e) => {
  const btn = e.target.closest('.rt-round-btn');
  if (!btn) return;
  rtState.picker = btn.dataset.picker === 'agent' ? 'agent' : 'user';
  rtPicker.querySelectorAll('.rt-round-btn').forEach(b => b.classList.toggle('active', b === btn));
});
