const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(
  path.join(__dirname, '../frontend/assets/js/roundtable.js'), 'utf8',
);
// Load the production renderer and event handler without attaching page listeners.
const logic = source.slice(0, source.indexOf('// 事件绑定'));

function setup() {
  const summary = {
    hidden: true, innerHTML: '',
    querySelectorAll: () => [],
    scrollIntoView: () => {},
  };
  const context = vm.createContext({
    rtSummary: summary,
    escapeHtml: value => String(value ?? '')
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
  });
  vm.runInContext(logic, context);
  return { context, summary };
}

test('renders the closing before positions and conditional applications', () => {
  const { context, summary } = setup();
  context._rtRenderSummary({
    closing: '双方仍保留分歧。',
    positions: [{ name: '甲', stance: '优先考虑就业。' }],
    applications: [{ name: '甲', scenario: '需要尽快就业时', application: '先评估专业的就业路径。' }],
  });
  assert.equal(summary.hidden, false);
  assert.match(summary.innerHTML, /主持人总结/);
  assert.match(summary.innerHTML, /保留分歧，不判胜负/);
  assert.ok(summary.innerHTML.indexOf('双方仍保留分歧。') < summary.innerHTML.indexOf('各方主张'));
  assert.match(summary.innerHTML, /观点的适用条件/);
  assert.match(summary.innerHTML, /需要尽快就业时/);
  assert.match(summary.innerHTML, /先评估专业的就业路径。/);
});

test('escapes closing and all application text instead of inserting HTML', () => {
  const { context, summary } = setup();
  const unsafe = '<img src=x onerror=alert(1)>';
  context._rtRenderSummary({
    closing: unsafe,
    applications: [{ name: unsafe, scenario: unsafe, application: unsafe }],
  });
  assert.ok(!summary.innerHTML.includes(unsafe));
  assert.equal(summary.innerHTML.split('&lt;img src=x onerror=alert(1)&gt;').length - 1, 4);
});

test('older summary payloads still render and do not invent an agreement or applications', () => {
  const { context, summary } = setup();
  context._rtRenderSummary({
    clashes: [{ a: '甲', b: '乙', point: '目标不同' }],
    positions: [{ name: '甲', stance: '重视稳定' }],
    open: ['如何权衡风险？'],
    speeches: [{ name: '甲', speeches: 1, quota: 2 }],
  });
  assert.match(summary.innerHTML, /目标不同/);
  assert.match(summary.innerHTML, /如何权衡风险？/);
  assert.match(summary.innerHTML, /甲 1\/2/);
  assert.ok(!summary.innerHTML.includes('rt-summary-closing'));
  assert.ok(!summary.innerHTML.includes('观点的适用条件'));
  assert.ok(!summary.innerHTML.includes('达成共识'));
  assert.match(summary.innerHTML, /按情境选择/);
});

test('degraded summaries preserve available positions without fabricated applicability', () => {
  const { context, summary } = setup();
  context._rtRenderSummary({
    closing: '发言已经结束。',
    positions: [{ name: '乙', stance: '保留原有主张' }],
    applications: [], degraded: true,
  });
  assert.match(summary.innerHTML, /保留原有主张/);
  assert.match(summary.innerHTML, /适用条件暂未整理/);
  assert.ok(!summary.innerHTML.includes('观点的适用条件'));
});

test('completion reasons describe discussion ending without implying agreement', async () => {
  const { context } = setup();
  const notices = [];
  context._rtNotice = text => notices.push(text);
  for (const reason of ['host', 'quota', 'rounds', 'budget']) {
    await context.handleRtEvent({ type: 'converged', reason });
  }
  assert.equal(notices.length, 4);
  assert.match(notices[0], /没有新的论点/);
  assert.match(notices[1], /发言额度已用完/);
  assert.match(notices[2], /交锋轮数已完成/);
  assert.match(notices[3], /调用已达上限/);
  assert.ok(notices.every(text => !/共识|一致|收敛/.test(text)));
});
