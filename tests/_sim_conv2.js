// 模拟验证前端对话管理逻辑（新对话复用 / _fromHistory 保护 / loadState 清理）
// 用法: node tests/_sim_conv2.js
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'frontend', 'index.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error('未找到 <script>'); process.exit(1); }

let code = m[1];
code += `
;globalThis.__getState = () => state;
;globalThis.__resetState = () => {
  state.conversations = [];
  state.currentConversation = null;
  state.currentCharacter = 'jung';
  state.currentScene = 'persona';
  state.characters = [];
};
;globalThis.__seedChars = () => {
  state.characters = [
    { id: 'jung', name: '荣格', avatar: 'J' },
    { id: 'adler', name: '阿德勒', avatar: 'A' },
  ];
};
;globalThis.__seedStorage = (data) => {
  const json = JSON.stringify(data);
  sessionStorage.setItem('multi_agent_rag_state', json);
  localStorage.setItem('multi_agent_rag_state', json);
};
;globalThis.__clearStorage = () => {
  sessionStorage.removeItem('multi_agent_rag_state');
  localStorage.removeItem('multi_agent_rag_state');
};
;globalThis.__metaMap = new Map();
;globalThis.__setMeta = (sid, char) => { globalThis.__metaMap.set(sid, char); };
`;

// ---------- mock DOM ----------
function makeClassList() {
  const s = new Set();
  return {
    toggle(c, f) { const on = f === undefined ? !s.has(c) : !!f; on ? s.add(c) : s.delete(c); return on; },
    contains(c) { return s.has(c); },
    add(c) { s.add(c); },
    remove(c) { s.delete(c); },
  };
}

function makeEl(id) {
  const el = {
    id, style: {}, dataset: {},
    _listeners: {},
    classList: makeClassList(),
    _innerHTML: '', textContent: '', value: '', placeholder: '',
    scrollHeight: 0, scrollTop: 0, disabled: false,
    focus() {}, appendChild() {},
    addEventListener(type, fn) { (el._listeners[type] ||= []).push(fn); },
    querySelectorAll() { return []; },
  };
  if (id === 'conversationList') {
    let _items = [];
    Object.defineProperty(el, 'innerHTML', {
      get() { return el._innerHTML; },
      set(v) {
        el._innerHTML = v;
        _items = [];
        const re = /data-conv-id="([^"]+)"/g;
        let mm;
        while ((mm = re.exec(v))) {
          const it = {
            dataset: { convId: mm[1] },
            _listeners: {},
            classList: makeClassList(),
            addEventListener(type, fn) { (it._listeners[type] ||= []).push(fn); },
          };
          _items.push(it);
        }
      },
    });
    el.querySelectorAll = (sel) => (sel === '.conv-item' ? _items : []);
  }
  if (id === 'charCards') {
    el._charCards = [];
    el.querySelectorAll = (sel) => (sel === '.char-card' ? el._charCards : []);
  }
  return el;
}

const elements = {};
const document = {
  getElementById(id) { return (elements[id] ||= makeEl(id)); },
  createElement() {
    const el = {
      style: {}, dataset: {}, classList: makeClassList(),
      textContent: '', innerHTML: '',
      _listeners: {},
      addEventListener(type, fn) { (el._listeners[type] ||= []).push(fn); },
      appendChild() {},
    };
    return el;
  },
  querySelector() { return { classList: makeClassList() }; },
};

const localStorage = { _m: new Map(), getItem(k) { return this._m.get(k) ?? null; }, setItem(k, v) { this._m.set(k, String(v)); }, removeItem(k) { this._m.delete(k); }, clear() { this._m.clear(); } };
const sessionStorage = { _m: new Map(), getItem(k) { return this._m.get(k) ?? null; }, setItem(k, v) { this._m.set(k, String(v)); }, removeItem(k) { this._m.delete(k); }, clear() { this._m.clear(); } };

let uuidSeq = 1;
const crypto = { randomUUID: () => 'uuid-' + (uuidSeq++) };
const fetchCalls = [];
const fetch = (url, opts) => {
  fetchCalls.push({ url: String(url), opts });
  if (String(url).includes('/persona/characters')) {
    return Promise.resolve({ ok: true, json: () => Promise.resolve({ characters: [{ id: 'jung', name: '荣格', avatar: 'J' }, { id: 'adler', name: '阿德勒', avatar: 'A' }] }) });
  }
  const mm = String(url).match(/^\/conversation\/([^/]+)\/meta$/);
  if (mm) {
    const c = sandbox.__metaMap.get(mm[1]);
    return Promise.resolve({ ok: true, json: () => Promise.resolve(c ? { known: true, character: c } : { known: false, character: null }) });
  }
  return Promise.resolve({ ok: true, json: () => Promise.resolve({ message: 'ok' }) });
};
const requestAnimationFrame = (fn) => fn();

const sandbox = { document, localStorage, sessionStorage, crypto, fetch, requestAnimationFrame, console, Date, JSON, Math, setTimeout, encodeURIComponent, decodeURIComponent };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

let pass = 0, fail = 0;
function assert(cond, msg) {
  if (cond) { pass++; console.log('  [PASS] ' + msg); }
  else { fail++; console.log('  [FAIL] ' + msg); }
}

function clickConv(convId) {
  const list = elements['conversationList'];
  const item = list.querySelectorAll('.conv-item').find(i => i.dataset.convId === convId);
  if (!item) throw new Error('未找到对话项 ' + convId);
  const fns = item._listeners['click'] || [];
  for (const fn of fns) fn({ target: { classList: { contains: () => false } } });
}

function seedCharCards() {
  const cc = elements['charCards'];
  cc._charCards.length = 0;
  cc._charCards.push({ dataset: { charId: 'jung' }, classList: makeClassList() });
  cc._charCards.push({ dataset: { charId: 'adler' }, classList: makeClassList() });
}

function isCharActive(charId) {
  const card = elements['charCards']._charCards.find(c => c.dataset.charId === charId);
  return card ? card.classList.contains('active') : false;
}

async function main() {
  const S = () => sandbox.__getState();

  // ---------- S1: 直接选角色（无历史） ----------
  console.log('\n[S1] 直接选角色（无历史对话）');
  sandbox.__resetState(); sandbox.__seedChars();
  sandbox.switchCharacter('adler');
  assert(S().conversations.length === 1, '创建 1 个新对话');
  assert(S().conversations[0].characterId === 'adler', '新对话角色 = adler');
  assert(S().currentConversation && S().currentConversation.id === S().conversations[0].id, '当前对话指向新对话');

  // ---------- S2: 点历史空对话 → 选角色（核心 bug） ----------
  console.log('\n[S2] 点历史空对话 → 选角色（不应重复创建空对话）');
  sandbox.__resetState(); sandbox.__seedChars();
  S().conversations = [{ id: 'c1', title: '新对话', messages: [], characterId: 'jung', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  assert(S().currentConversation && S().currentConversation.id === 'c1', '点击后当前对话 = c1');
  assert(S().currentConversation._fromHistory === true, 'c1 被打上 _fromHistory 标记');
  const lenBefore = S().conversations.length;
  sandbox.switchCharacter('adler');
  assert(S().conversations.length === lenBefore, '★ 核心：没有创建第二个对话（复用已有空对话）');
  assert(S().conversations[0]._fromHistory === undefined, '复用后 _fromHistory 标记被清除');
  assert(S().conversations[0].characterId === 'adler', '空对话归属改写为 adler');
  assert(S().currentConversation.id === 'c1', '当前对话仍是 c1（未新建）');

  // ---------- S3: 点历史有消息对话 → 选角色（回归：历史归属不被改写） ----------
  console.log('\n[S3] 点历史有消息对话 → 选角色（历史归属不被改写）');
  sandbox.__resetState(); sandbox.__seedChars();
  S().conversations = [{ id: 'c1', title: '与荣格对话', messages: [{ type: 'user', content: 'hi', extra: {} }], characterId: 'jung', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  sandbox.switchCharacter('adler');
  assert(S().conversations.length === 2, '创建新对话（历史对话有内容，不可复用）');
  assert(S().conversations[0].characterId === 'jung', '历史对话归属保持 jung（不被改写）');
  assert(S().conversations[1].characterId === 'adler', '新对话角色 = adler');
  assert(S().currentConversation.id === S().conversations[1].id, '当前对话为新对话');

  // ---------- S4: 已有空对话 A + 有消息 B → 点 B → 选角色 → 复用 A ----------
  console.log('\n[S4] 已有空对话A + 有消息B，点B后选角色（应复用A）');
  sandbox.__resetState(); sandbox.__seedChars();
  S().conversations = [
    { id: 'A', title: '新对话', messages: [], characterId: 'jung', scene: 'persona', sessionId: 'sA', createdAt: Date.now() },
    { id: 'B', title: '与荣格对话', messages: [{ type: 'user', content: 'hi', extra: {} }], characterId: 'jung', scene: 'persona', sessionId: 'sB', createdAt: Date.now() },
  ];
  sandbox.renderConversations();
  clickConv('B');
  sandbox.switchCharacter('adler');
  assert(S().conversations.length === 2, '复用空对话 A，不创建新对话');
  assert(S().conversations[0].id === 'A' && S().conversations[0].characterId === 'adler', 'A 改写为 adler');
  assert(S().conversations[1].id === 'B' && S().conversations[1].characterId === 'jung', 'B 保持 jung');
  assert(S().currentConversation.id === 'A', '当前对话 = A');

  // ---------- S5: loadState 清除持久化的 _fromHistory ----------
  console.log('\n[S5] 刷新后 loadState 清除 _fromHistory 残留');
  sandbox.__resetState(); sandbox.__seedChars();
  sandbox.__seedStorage({
    conversations: [{ id: 'c1', title: '新对话', messages: [], characterId: 'jung', scene: 'persona', sessionId: 's1', createdAt: Date.now(), _fromHistory: true }],
    currentConversationId: 'c1', currentScene: 'persona', currentCharacter: 'jung', sessionId: 's1',
  });
  const ok = sandbox.loadState();
  assert(ok === true, 'loadState 返回 true');
  assert(S().conversations[0]._fromHistory === undefined, '_fromHistory 残留被清除');
  sandbox.switchCharacter('adler');
  assert(S().conversations.length === 1, '选角色后复用原对话，无新增');
  assert(S().conversations[0].characterId === 'adler', '对话改写为 adler');

  // ---------- S6: 连续切换角色 ----------
  console.log('\n[S6] 连续切换角色（始终只有一个对话）');
  sandbox.__resetState(); sandbox.__seedChars();
  sandbox.switchCharacter('adler');
  sandbox.switchCharacter('jung');
  sandbox.switchCharacter('adler');
  assert(S().conversations.length === 1, '始终只有 1 个对话');

  // ---------- S7: 污染数据（characterId 被改写但消息记录是荣格）→ 点击应高亮荣格 ----------
  console.log('\n[S7] 污染数据：characterId=adler 但消息实际是荣格 → 点击后角色栏亮荣格');
  sandbox.__resetState(); sandbox.__seedChars();
  seedCharCards();
  S().conversations = [{ id: 'c1', title: '与荣格对话', messages: [{ type: 'assistant', content: '你好', extra: { sceneIcon: 'J' } }], characterId: 'adler', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  assert(S().currentCharacter === 'jung', 'currentCharacter 修正为 jung（按消息记录判断）');
  assert(isCharActive('jung'), '角色栏：荣格卡片高亮');
  assert(!isCharActive('adler'), '角色栏：阿德勒卡片不高亮');

  // ---------- S8: 点击历史对话 → 角色栏高亮对应角色 ----------
  console.log('\n[S8] 正常数据：点击阿德勒对话 → 角色栏亮阿德勒');
  sandbox.__resetState(); sandbox.__seedChars();
  seedCharCards();
  S().conversations = [
    { id: 'c1', title: '与荣格对话', messages: [{ type: 'assistant', content: 'hi', extra: { sceneIcon: 'J' } }], characterId: 'jung', scene: 'persona', sessionId: 's1', createdAt: Date.now() },
    { id: 'c2', title: '与阿德勒对话', messages: [{ type: 'assistant', content: 'hi', extra: { sceneIcon: 'A' } }], characterId: 'adler', scene: 'persona', sessionId: 's2', createdAt: Date.now() },
  ];
  sandbox.renderConversations();
  clickConv('c1');
  assert(isCharActive('jung') && !isCharActive('adler'), '点荣格对话 → 荣格亮、阿德勒不亮');
  clickConv('c2');
  assert(isCharActive('adler') && !isCharActive('jung'), '再点阿德勒对话 → 阿德勒亮、荣格不亮');

  // ---------- S9: 空对话点击 → 按 characterId 回退高亮 ----------
  console.log('\n[S9] 空对话（无消息）→ 按 characterId 回退高亮');
  sandbox.__resetState(); sandbox.__seedChars();
  seedCharCards();
  S().conversations = [{ id: 'c1', title: '新对话', messages: [], characterId: 'adler', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  assert(S().currentCharacter === 'adler', '空对话按 characterId 回退 → adler');
  assert(isCharActive('adler'), '角色栏：阿德勒卡片高亮');

  // ---------- S10: 无 sceneIcon + characterId 被污染 → 后端 meta 权威修正（核心场景） ----------
  console.log('\n[S10] 污染数据无消息证据 + 后端记录 → 点击历史对话后角色修正并自愈');
  sandbox.__resetState(); sandbox.__seedChars();
  seedCharCards();
  sandbox.__setMeta('s1', 'jung');
  S().conversations = [{ id: 'c1', title: '与荣格对话', messages: [{ type: 'assistant', content: '你好', extra: {} }], characterId: 'adler', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  await sleep(10);
  assert(S().currentCharacter === 'jung', '★ 点击后角色修正为荣格（后端权威优先）');
  assert(S().conversations[0].characterId === 'jung', 'characterId 自愈为 jung');
  assert(isCharActive('jung') && !isCharActive('adler'), '角色栏：荣格亮、阿德勒不亮');
  assert(S().conversations[0].messages[0].extra.sceneIcon === 'J', '消息 sceneIcon 补写为荣格头像');

  // ---------- S11: 后端无记录 → 保持本地判断（不崩、不误改） ----------
  console.log('\n[S11] 后端无记录 → 保持本地判断');
  sandbox.__resetState(); sandbox.__seedChars();
  seedCharCards();
  sandbox.__setMeta('s1', null);
  S().conversations = [{ id: 'c1', title: '与荣格对话', messages: [{ type: 'assistant', content: '你好', extra: { sceneIcon: 'J' } }], characterId: 'adler', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  await sleep(10);
  assert(S().currentCharacter === 'jung', '按消息 sceneIcon 判断为荣格');
  assert(S().conversations[0].characterId === 'adler', '后端无记录时 characterId 不被误改');

  // ---------- S12: 本地与后端一致 → 无多余改动 ----------
  console.log('\n[S12] 本地与后端一致 → 无多余改动');
  sandbox.__resetState(); sandbox.__seedChars();
  seedCharCards();
  sandbox.__setMeta('s1', 'jung');
  S().conversations = [{ id: 'c1', title: '与荣格对话', messages: [{ type: 'assistant', content: '你好', extra: { sceneIcon: 'J' } }], characterId: 'jung', scene: 'persona', sessionId: 's1', createdAt: Date.now() }];
  sandbox.renderConversations();
  clickConv('c1');
  await sleep(10);
  assert(S().currentCharacter === 'jung', '点击后角色 = 荣格');
  assert(S().conversations[0].characterId === 'jung', 'characterId 保持不变（一致）');
  assert(S().conversations[0].messages[0].extra.sceneIcon === 'J', 'sceneIcon 保持不变（一致）');

  console.log('\n========================================');
  console.log('RESULT: ' + pass + ' passed, ' + fail + ' failed');
  process.exit(fail > 0 ? 1 : 0);
}

main().catch(e => { console.error('脚本异常:', e); process.exit(2); });
