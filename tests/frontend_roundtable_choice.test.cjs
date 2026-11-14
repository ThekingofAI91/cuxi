const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../frontend/assets/js/roundtable.js'), 'utf8');

// Minimal DOM: exercise the production choice handlers and fetch/SSE boundary
// without a browser, API server, model calls, or wall-clock countdowns.
class Element {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.children = [];
    this.listeners = {};
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.textContent = '';
    this.queries = new Map();
    const classes = new Set();
    this.classList = {
      add(...names) { names.forEach(name => classes.add(name)); },
      remove(...names) { names.forEach(name => classes.delete(name)); },
      toggle(name, on) { if (on ?? !classes.has(name)) classes.add(name); else classes.delete(name); },
      contains(name) { return classes.has(name); },
    };
  }
  set innerHTML(value) {
    this.html = value; this.children = []; this.queries = new Map();
    const quota = value.match(/class="rt-seat-quota"[^>]*>(.*?)<\/span>/s);
    if (quota) this.queries.set('.rt-seat-quota i', [...quota[1].matchAll(/<i>/g)].map(() => new Element('i')));
  }
  get innerHTML() { return this.html || ''; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  click() { if (!this.disabled && this.listeners.click) return this.listeners.click({ target: this }); }
  setAttribute(name, value) { this[name] = value; }
  querySelector(selector) {
    if (!this.queries.has(selector)) this.queries.set(selector, new Element());
    return this.queries.get(selector);
  }
  querySelectorAll(selector) { return this.queries.get(selector) || []; }
  scrollIntoView() {}
}

function sse(events, readError) {
  let read = false;
  return {
    ok: true,
    body: { getReader: () => ({
      async read() {
        if (readError) throw new Error(readError);
        if (read) return { done: true };
        read = true;
        return { done: false, value: new TextEncoder().encode(
          events.map(event => `data: ${JSON.stringify(event)}\n\n`).join(''),
        ) };
      },
      async cancel() {},
    }) },
  };
}

function request(id = 'a', decisionId = `decision-${id}`) {
  const bid = { id, name: id.toUpperCase(), speeches: 1, quota: 4, urgency: 4,
    move: 'rebut', reason: '我要回应这项判断。', target: { speaker: '乙', quote: '只能如此。' } };
  return { type: 'choice_request', decision_id: decisionId, round: 1, total: 3, candidates: [bid],
    intents: [{ ...bid, character_id: id, status: 'ok', want: true }] };
}

function snapshot(overrides = {}) {
  return {
    session_id: 'session', topic: '议题', picker: 'user', rounds: 3, round_index: 1,
    speakers: [{ id: 'a', name: 'A', quota: 4 }, { id: 'b', name: 'B', quota: 4 }],
    ledger: { a: { speeches: 1, quota: 4 }, b: { speeches: 1, quota: 4 } },
    transcript: [
      { turn_id: 'a-opening', character_id: 'a', name: 'A', kind: 'opening', round: 0, content: '甲的完整开场。' },
      { turn_id: 'b-opening', character_id: 'b', name: 'B', kind: 'opening', round: 0, content: '乙的完整开场。' },
    ],
    status: 'waiting', pending: true, choice_request: request('b'), ...overrides,
  };
}

function setup(respond = () => sse([{ type: 'end' }]), storage = new Map()) {
  const elements = new Map();
  const el = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const calls = [];
  const notices = [];
  const context = vm.createContext({
    TextDecoder,
    crypto: { randomUUID: require('node:crypto').randomUUID },
    document: { createElement: tag => new Element(tag), getElementById: el },
    rtAsk: el('rtAsk'), rtAskList: el('rtAskList'), rtAskCancel: el('rtAskCancel'),
    rtBackBtn: null, rtRounds: null, rtPicker: null, rtStage: el('rtStage'), rtSummary: el('rtSummary'),
    rtStartBtn: el('rtStartBtn'), rtTopic: el('rtTopic'), rtSetup: el('rtSetup'),
    rtTranscript: el('rtTranscript'), roundtableView: el('roundTableView'),
    rtChips: el('rtChips'), homeView: el('homeView'),
    state: { characters: [{ id: 'a', name: 'A' }, { id: 'b', name: 'B' }] }, RT_MAX: 3,
    closeSidebar() {}, homeStage() {}, showHome() {},
    nameMark: value => String(value || '').slice(0, 1),
    renderMarkdownLite: value => String(value).replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    localStorage: {
      getItem: key => storage.get(key) || null,
      setItem: (key, value) => storage.set(key, value),
      removeItem: key => storage.delete(key),
    },
    setInterval() { throw new Error('Choice must not schedule a countdown'); },
    setTimeout() { throw new Error('Choice must not auto-select later'); },
    escapeHtml: value => String(value ?? '').replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    async fetch(url, options) {
      calls.push({ url, options });
      return respond(url, options, calls.length);
    },
  });
  vm.runInContext(source, context);
  vm.runInContext("rtState.sessionId = 'session'; rtState.selected = new Set(['a', 'b']);", context);
  context._rtBarText = () => {};
  context._rtScroll = () => {};
  context._rtNotice = text => notices.push(text);
  const choices = () => el('rtAskList').children.flatMap(card => card.children);
  const speeches = () => el('rtTranscript').children.filter(card => (card.className || '').startsWith('rt-speaker '));
  return { context, el, calls, choices, notices, storage, speeches };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

test('a single willing speaker waits for an explicit click, with no countdown', async () => {
  const { context, el, calls, choices } = setup();
  let finished = false;
  const waiting = context.handleRtEvent(request()).then(() => { finished = true; });
  await flush();
  assert.equal(el('rtAsk').hidden, false);
  assert.equal(finished, false);
  assert.equal(calls.length, 0);
  assert.equal(choices().length, 1);
  choices()[0].click();
  await waiting;
  assert.deepEqual(JSON.parse(calls[0].options.body), { character_id: 'a', decision_id: 'decision-a' });
});

test('starting and resuming a meeting waits for a fresh user choice in every round', async () => {
  const { context, el, calls, choices } = setup((url, options, count) => {
    if (count === 1) return sse([{ type: 'start', session_id: 'session', speakers: [] }, request('a')]);
    if (count === 2) return sse([{ ...request('b'), round: 2 }]);
    return sse([{ type: 'end' }]);
  });
  el('rtTopic').value = '议题';
  const meeting = context.startRoundtable();
  await flush();
  assert.equal(calls.length, 1);
  assert.equal(vm.runInContext('rtState.loading', context), true);
  choices()[0].click();
  await flush();
  assert.equal(calls.length, 2);
  assert.equal(el('rtAsk').hidden, false);
  assert.equal(choices()[0].dataset.id, 'b');
  assert.equal(vm.runInContext('rtState.loading', context), true);
  choices()[0].click();
  await meeting;
  assert.equal(calls.length, 3);
  assert.equal(vm.runInContext('rtState.loading', context), false);
  assert.equal(el('rtTopic').disabled, false);
});

for (const failure of ['fetch response lost', 'start SSE lost']) {
  test(`an accepted request with ${failure} recovers by the session ID saved before POST`, async () => {
    let issuedSession;
    const harness = setup((url, options, count) => {
      if (count === 1) {
        issuedSession = JSON.parse(options.body).session_id;
        assert.match(issuedSession, /^[0-9a-f-]{36}$/i);
        assert.equal(harness.context._rtSavedSession().session_id, issuedSession);
        if (failure === 'fetch response lost') throw new Error('回执丢失');
        return sse([], 'start 回执丢失');
      }
      if (!options) return { ok: true, json: async () => snapshot({ session_id: issuedSession }) };
      return sse([{ type: 'end', session_id: issuedSession }]);
    });
    harness.el('rtTopic').value = '新议题';
    const meeting = harness.context.startRoundtable();
    await flush();
    assert.equal(harness.calls.length, 2);
    assert.equal(harness.calls[1].url, `/persona/roundtable/${issuedSession}/snapshot`);
    assert.equal(harness.speeches().length, 2);
    assert.equal(harness.el('rtAsk').hidden, false);
    harness.el('rtAskCancel').click();
    await meeting;
    assert.equal(harness.calls[2].url, `/persona/roundtable/${issuedSession}/choice`);
    assert.equal(harness.context._rtSavedSession().session_id, issuedSession);
  });
}

for (const status of [400, 429]) {
  test(`a rejected creation (${status}) restores the previous real meeting entry`, async () => {
    const { context, el, calls } = setup(() => ({ ok: false, status, json: async () => ({ detail: '创建未被接受' }) }));
    context._rtRememberSession('previous-real-session', '上一场', 'waiting');
    el('rtTopic').value = '新议题';
    await context.startRoundtable();
    assert.equal(calls.length, 1);
    assert.notEqual(JSON.parse(calls[0].options.body).session_id, 'previous-real-session');
    assert.equal(context._rtSavedSession().session_id, 'previous-real-session');
    assert.equal(context._rtSavedSession().topic, '上一场');
    assert.equal(el('rtResumeEntry').hidden, false);
    assert.equal(vm.runInContext('rtState.sessionId', context), null);
  });
}

test('a rejected first creation removes only its own uncreated local entry', async () => {
  const { context, el, calls } = setup(() => ({ ok: false, status: 422, json: async () => ({ detail: '参数不符' }) }));
  el('rtTopic').value = '新议题';
  await context.startRoundtable();
  assert.equal(calls.length, 1);
  assert.equal(context._rtSavedSession(), null);
  assert.equal(el('rtResumeEntry').hidden, true);
});

test('full receipts are escaped, while unwilling and failed speakers cannot be selected', async () => {
  const { context, el, choices } = setup();
  const unsafe = '<img src=x onerror=alert(1)>';
  const bid = request().candidates[0];
  Object.assign(bid, { name: unsafe, reason: unsafe + '\n第二行', move: 'distinguish',
    target: { speaker: unsafe, quote: unsafe }, parse_error: unsafe });
  const waiting = context._rtAskShow({ candidates: [bid], intents: [
    { ...bid, character_id: 'a', want: true, status: 'ok' },
    { character_id: 'b', name: '乙', want: false, reason: '目前没有补充。', status: 'ok' },
    { character_id: 'c', name: '丙', want: false, error: unsafe, status: 'error' },
  ] });
  const cards = el('rtAskList').children;
  assert.equal(cards.length, 3);
  assert.equal(choices().length, 1);
  assert.equal(choices()[0].dataset.id, 'a'); // A parse fallback alone is not a failed bid.
  assert.equal(choices()[0].textContent, `请${unsafe}发言`);
  assert.ok(cards.every(card => !card.innerHTML.includes(unsafe)));
  assert.match(cards[0].innerHTML, /紧迫度 4\/5 · 提出区分 · 已发言 1\/4 次/);
  assert.match(cards[0].innerHTML, /第二行/);
  assert.match(cards[0].innerHTML, /回应对象：&lt;img/);
  assert.match(cards[0].innerHTML, /rt-ask-quote/);
  assert.match(cards[1].innerHTML, /暂不发言/);
  assert.match(cards[1].innerHTML, /目前没有补充/);
  assert.match(cards[2].innerHTML, /请缨失败/);
  context._rtAskResolve(null);
  await waiting;
});

test('no willing speakers still waits for the user to end the discussion', async () => {
  const { context, el, calls, choices, notices } = setup(() => sse([
    { type: 'converged', reason: 'user' }, { type: 'end' },
  ]));
  const waiting = context.handleRtEvent({ type: 'choice_request', decision_id: 'empty-round', round: 2,
    candidates: [], intents: [{ character_id: 'a', name: '甲', want: false, reason: '没有补充。' }] });
  await flush();
  assert.equal(calls.length, 0);
  assert.equal(choices().length, 0);
  assert.equal(el('rtAskTitle').textContent, '本轮无人请缨');
  el('rtAskCancel').click();
  await waiting;
  assert.deepEqual(JSON.parse(calls[0].options.body), { character_id: '__end__', decision_id: 'empty-round' });
  assert.deepEqual(notices, ['你已结束本场讨论。']);
});

for (const failure of ['HTTP failure', 'SSE read failure', 'premature SSE close']) {
  test(`${failure} restores the current pending choices without resubmitting automatically`, async () => {
    const { context, el, calls, choices } = setup((url, options, count) => {
      if (count === 1) {
        if (failure === 'HTTP failure') return { ok: false, status: 503, json: async () => ({ detail: '服务暂不可用' }) };
        if (failure === 'SSE read failure') return sse([], '网络中断');
        return sse([{ type: 'token', content: '未完成' }]);
      }
      if (url.endsWith('/snapshot')) return { ok: true, json: async () => snapshot() };
      return sse([{ type: 'end' }]);
    });
    const waiting = context.handleRtEvent(request());
    choices()[0].click();
    await flush();
    assert.equal(calls.length, 2);
    assert.ok(calls[1].url.endsWith('/snapshot'));
    assert.equal(el('rtAsk').hidden, false);
    assert.equal(el('rtAskError').hidden, false);
    assert.match(el('rtAskError').textContent, /未能接续讨论/);
    assert.equal(choices()[0].dataset.id, 'b');
    choices()[0].click();
    await waiting;
    assert.deepEqual(JSON.parse(calls[2].options.body), { character_id: 'b', decision_id: 'decision-b' });
  });
}

test('failed recovery offers a manual reconnect before allowing another decision', async () => {
  const { context, el, calls, choices } = setup((url, options, count) => {
    if (count <= 2) throw new Error('暂时离线');
    if (url.endsWith('/snapshot')) return { ok: true, json: async () => snapshot() };
    return sse([{ type: 'end' }]);
  });
  const waiting = context.handleRtEvent(request());
  choices()[0].click();
  await flush();
  assert.equal(calls.length, 2);
  assert.equal(el('rtRecovery').hidden, false);
  assert.equal(el('rtAsk').hidden, true);
  assert.equal(calls.length, 2);
  el('rtRefreshBtn').click();
  await flush();
  assert.equal(calls.length, 3);
  assert.equal(el('rtRecovery').hidden, true);
  assert.equal(el('rtAskCancel').disabled, false);
  choices()[0].click();
  await waiting;
  assert.deepEqual(JSON.parse(calls[3].options.body), { character_id: 'b', decision_id: 'decision-b' });
});

test('cleanup, leaving the view, and empty choices never submit or end a meeting', async () => {
  const { context, calls, el } = setup();
  await context._rtSendChoice(null);
  await context._rtSendChoice('');
  const waiting = context.handleRtEvent(request());
  context.closeRoundtable();
  await waiting;
  assert.equal(calls.length, 0);
  assert.equal(el('rtAsk').hidden, true);
  assert.equal(el('roundTableView').hidden, true);
});

test('a deleted snapshot gives visible feedback and restores setup', async () => {
  const { context, el, choices, notices } = setup((url, options) => options
    ? { ok: false, status: 409, json: async () => ({ detail: '会议已结束' }) }
    : { ok: false, status: 404 });
  el('rtSetup').hidden = true;
  const waiting = context.handleRtEvent(request());
  choices()[0].click();
  await waiting;
  assert.equal(el('rtSetup').hidden, false);
  assert.match(notices[0], /会议记录已不存在/);
});

test('a saved meeting survives a new page context and resumes from its full snapshot', async () => {
  const storage = new Map();
  const first = setup(undefined, storage);
  await first.context.handleRtEvent({ type: 'start', session_id: 'saved-session', topic: '保留这个议题', speakers: [] });
  first.context.closeRoundtable();
  const restored = setup((url, options) => options
    ? sse([{ type: 'end' }])
    : { ok: true, json: async () => snapshot({ session_id: 'saved-session', topic: '保留这个议题' }) }, storage);
  restored.context.openRoundtable();
  assert.equal(restored.el('rtResumeEntry').hidden, false);
  assert.equal(restored.el('rtResumeBtn').textContent, '继续会议');
  assert.match(restored.el('rtResumeLabel').textContent, /保留这个议题/);
  assert.equal(restored.calls.length, 0);
  const resuming = restored.el('rtResumeBtn').click();
  await flush();
  assert.equal(restored.calls[0].url, '/persona/roundtable/saved-session/snapshot');
  assert.equal(restored.speeches().length, 2);
  assert.equal(restored.el('rtAsk').hidden, false);
  assert.equal(restored.el('rtTopic').value, '保留这个议题');
  assert.equal(restored.calls.length, 1);
  restored.el('rtAskLeaveBtn').click();
  await resuming;
  assert.equal(restored.calls.length, 1); // Leaving keeps the checkpoint, without ending or choosing.
  assert.equal(restored.context._rtSavedSession().session_id, 'saved-session');
});

test('snapshot reconciliation replaces partial text, restores four-slot quotas, and does not duplicate turns', async () => {
  const { context, speeches } = setup();
  await context.handleRtEvent({ type: 'speaker_start', character_id: 'a', name: 'A', speeches: 1, quota: 4 });
  await context.handleRtEvent({ type: 'token', content: '只收到半句' });
  const complete = snapshot({
    ledger: { a: { speeches: 3, quota: 4 }, b: { speeches: 1, quota: 4 } },
    transcript: [...snapshot().transcript,
      { turn_id: 'a-1', speaker_id: 'a', name: 'A', kind: 'rebuttal', round: 1, content: '浏览器漏收的完整发言。' },
      { turn_id: 'a-2', character_id: 'a', name: 'A', kind: 'rebuttal', round: 2, content: '下一段完整发言。' },
    ],
  });
  context._rtApplySnapshot(complete);
  assert.equal(speeches().length, 4);
  assert.deepEqual(speeches().map(card => card.querySelector('.rt-speaker-text').innerHTML), [
    '甲的完整开场。', '乙的完整开场。', '浏览器漏收的完整发言。', '下一段完整发言。',
  ]);
  const seats = vm.runInContext('rtState._stage', context);
  const dots = seats.get('a').querySelectorAll('.rt-seat-quota i');
  assert.equal(dots.length, 4);
  assert.equal(dots.filter(dot => dot.classList.contains('on')).length, 3);
  context._rtApplySnapshot(complete);
  assert.equal(speeches().length, 4);
  assert.equal(speeches()[2].dataset.turnId, 'a-1');
  await context.handleRtEvent({ type: 'end', ...complete, transcript: [...complete.transcript,
    { turn_id: 'b-3', character_id: 'b', name: 'B', kind: 'rebuttal', round: 3, content: '结束时补回的最后发言。' },
  ] });
  assert.equal(speeches().length, 5);
  assert.equal(speeches()[4].querySelector('.rt-speaker-text').innerHTML, '结束时补回的最后发言。');
});

test('409 reloads the full transcript and only submits a newly clicked decision token', async () => {
  const recovered = snapshot({ choice_request: request('b', 'new-decision'), transcript: [...snapshot().transcript,
    { turn_id: 'a-1', character_id: 'a', name: 'A', kind: 'rebuttal', round: 1, content: '前一次点将已经完成。' },
  ] });
  const { context, calls, choices, speeches } = setup((url, options, count) => {
    if (count === 1) return { ok: false, status: 409, json: async () => ({ detail: '旧决定已处理' }) };
    if (!options) return { ok: true, json: async () => recovered };
    return sse([{ type: 'end' }]);
  });
  const waiting = context.handleRtEvent(request('a', 'old-decision'));
  choices()[0].click();
  await flush();
  assert.equal(calls.length, 2);
  assert.equal(speeches().length, 3);
  assert.equal(speeches()[2].querySelector('.rt-speaker-text').innerHTML, '前一次点将已经完成。');
  assert.equal(choices()[0].dataset.id, 'b');
  choices()[0].click();
  await waiting;
  assert.deepEqual(calls.filter(call => call.options).map(call => JSON.parse(call.options.body)), [
    { character_id: 'a', decision_id: 'old-decision' },
    { character_id: 'b', decision_id: 'new-decision' },
  ]);
});

test('an accepted but disconnected request is not repeated while the snapshot is running', async () => {
  const { context, el, calls, choices } = setup((url, options, count) => {
    if (count === 1) throw new Error('响应断开');
    if (count === 2) return { ok: true, json: async () => snapshot({ status: 'running', pending: false, choice_request: null }) };
    if (count === 3) return { ok: true, json: async () => snapshot({ choice_request: request('b', 'after-running') }) };
    return sse([{ type: 'end' }]);
  });
  const waiting = context.handleRtEvent(request());
  choices()[0].click();
  await flush();
  assert.equal(calls.length, 2);
  assert.equal(el('rtAsk').hidden, true);
  assert.equal(el('rtRecovery').hidden, false);
  assert.equal(el('rtContinueBtn').hidden, true);
  assert.match(el('rtRecoveryText').textContent, /会议正在生成/);
  await flush();
  assert.equal(calls.length, 2);
  el('rtRefreshBtn').click();
  await flush();
  assert.equal(calls.length, 3);
  choices()[0].click();
  await waiting;
  assert.deepEqual(JSON.parse(calls[3].options.body), { character_id: 'b', decision_id: 'after-running' });
});

test('interrupted meetings continue only on request and consume the snapshot before the resumed SSE', async () => {
  const interrupted = snapshot({ status: 'interrupted', pending: false, choice_request: null });
  let completed = false;
  const { context, el, calls, choices, speeches } = setup((url, options) => {
    if (!options) return { ok: true, json: async () => completed
      ? snapshot({ status: 'completed', pending: false, choice_request: null }) : interrupted };
    if (url.endsWith('/continue')) return sse([{ type: 'snapshot', data: interrupted }, request('b', 'continued-decision')]);
    completed = true;
    return sse([{ type: 'end' }]);
  });
  context._rtRememberSession('session', '议题', 'running');
  const resuming = context.resumeRoundtable();
  await flush();
  assert.equal(calls.length, 1);
  assert.equal(el('rtContinueBtn').hidden, false);
  assert.equal(speeches().length, 2);
  el('rtContinueBtn').click();
  await flush();
  assert.equal(calls[1].url, '/persona/roundtable/session/continue');
  assert.equal(calls[1].options.body, '{}');
  assert.equal(speeches().length, 2);
  assert.equal(choices()[0].dataset.id, 'b');
  el('rtAskCancel').click();
  await resuming;
  assert.deepEqual(JSON.parse(calls[2].options.body), { character_id: '__end__', decision_id: 'continued-decision' });
});

test('a snapshot-only continue response rechecks a running meeting without reporting a broken stream', async () => {
  const interrupted = snapshot({ status: 'interrupted', pending: false, choice_request: null });
  const running = snapshot({ status: 'running', pending: false, choice_request: null });
  const { context, el, calls } = setup((url, options, count) => {
    if (count === 1) return { ok: true, json: async () => interrupted };
    if (count === 2) return sse([{ type: 'snapshot', data: running }]);
    if (count === 3) return { ok: true, json: async () => running };
    if (count === 4) return { ok: true, json: async () => snapshot({ choice_request: request('b', 'after-race') }) };
    return sse([{ type: 'end' }]);
  });
  context._rtRememberSession('session', '议题', 'running');
  const resuming = context.resumeRoundtable();
  await flush();
  el('rtContinueBtn').click();
  await flush();
  assert.equal(calls.length, 3);
  assert.equal(calls[1].url, '/persona/roundtable/session/continue');
  assert.equal(calls[2].url, '/persona/roundtable/session/snapshot');
  assert.equal(el('rtRecovery').hidden, false);
  assert.match(el('rtRecoveryText').textContent, /会议正在生成/);
  assert.doesNotMatch(el('rtRecoveryText').textContent, /中断|未能接续/);
  assert.equal(calls.filter(call => call.url.endsWith('/choice')).length, 0);
  el('rtRefreshBtn').click();
  await flush();
  assert.equal(calls.length, 4);
  el('rtAskCancel').click();
  await resuming;
  assert.deepEqual(JSON.parse(calls[4].options.body), { character_id: '__end__', decision_id: 'after-race' });
});

test('a completed snapshot restores its body without leaving a nonexistent summary loading', async () => {
  const { context, el, calls, speeches } = setup(() => ({ ok: true, json: async () => snapshot({
    status: 'completed', pending: false, choice_request: null, summary: {},
  }) }));
  context._rtRememberSession('session', '议题', 'waiting');
  await context.resumeRoundtable();
  assert.equal(calls.length, 1);
  assert.equal(speeches().length, 2);
  assert.equal(el('rtSummary').hidden, true);
  assert.equal(el('rtSetup').hidden, false);
  assert.equal(el('rtResumeBtn').textContent, '查看上次会议');
});

test('double-clicking a choice sends one request, and a missing decision token sends none', async () => {
  const { context, calls, choices } = setup();
  await assert.rejects(context._rtSendChoice('a', ''), /点将信息已失效/);
  assert.equal(calls.length, 0);
  const waiting = context.handleRtEvent(request());
  const button = choices()[0];
  button.click();
  button.click();
  await waiting;
  assert.equal(calls.length, 1);
});
