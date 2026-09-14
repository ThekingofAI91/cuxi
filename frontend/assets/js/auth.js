"use strict";
// ============================================================
// 账号：登录 / 注册 / 会话态
// ============================================================
// 会话态由后端 HttpOnly Cookie 承载（前端 JS 不可读，防 XSS 偷 token）。
// 前端只负责：弹层交互、提交表单、把 /auth/me 的结果显示在侧栏按钮上。
// 未登录完全不影响使用——登录是可选能力。

const authModal = document.getElementById('authModal');
const authBtn = document.getElementById('authBtn');
const authCloseBtn = document.getElementById('authCloseBtn');
const authSubmitBtn = document.getElementById('authSubmitBtn');
const authStatus = document.getElementById('authStatus');

let _authMode = 'login'; // 'login' | 'register'
let _authBusy = false;

function _setAuthStatus(msg, ok) {
  if (!authStatus) return;
  authStatus.textContent = msg || '';
  authStatus.className = 'auth-status' + (ok ? ' ok' : '');
}

function _switchAuthTab(mode) {
  _authMode = mode;
  document.querySelectorAll('.auth-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.authTab === mode));
  document.getElementById('authLoginForm').hidden = mode !== 'login';
  document.getElementById('authRegisterForm').hidden = mode !== 'register';
  if (authSubmitBtn) authSubmitBtn.textContent = mode === 'login' ? '登录' : '注册并登录';
  _setAuthStatus('', true);
}

function openAuthModal() {
  if (!authModal) return;
  _switchAuthTab(_authMode);
  authModal.hidden = false;
  _setAuthStatus('', true);
  (document.getElementById('authLoginAccount') || {}).focus?.();
}

function closeAuthModal() {
  if (authModal) authModal.hidden = true;
}

// 侧栏入口按钮文案：未登录显示"登录 / 注册"，已登录显示"昵称 · 退出"
function renderAuthEntry(user) {
  if (!authBtn) return;
  const delBtn = document.getElementById('authDeleteBtn');
  if (delBtn) delBtn.hidden = !user;   // 注销账号入口仅登录后可见（需凭会话操作）
  if (user) {
    authBtn.textContent = `${user.display_name || user.account} · 退出`;
    authBtn.title = '点击退出登录';
    authBtn.dataset.loggedIn = '1';
  } else {
    authBtn.textContent = '登录 / 注册';
    authBtn.title = '登录后额度按账号计算';
    authBtn.dataset.loggedIn = '';
  }
}

async function refreshAuthState() {
  try {
    const resp = await fetch('/auth/me');
    const data = await resp.json();
    renderAuthEntry(data.user || null);
  } catch (e) { /* 服务未连接时保持默认文案 */ }
}

async function submitAuth() {
  if (_authBusy) return;
  const isLogin = _authMode === 'login';
  const account = (document.getElementById(isLogin ? 'authLoginAccount' : 'authRegAccount')?.value || '').trim();
  const password = document.getElementById(isLogin ? 'authLoginPassword' : 'authRegPassword')?.value || '';
  const displayName = isLogin ? '' : (document.getElementById('authRegName')?.value || '').trim();

  if (!account) { _setAuthStatus('请填写账号'); return; }
  if (!password) { _setAuthStatus('请填写密码'); return; }

  _authBusy = true;
  if (authSubmitBtn) { authSubmitBtn.disabled = true; }
  _setAuthStatus(isLogin ? '登录中…' : '注册中…', true);
  try {
    const resp = await fetch(isLogin ? '/auth/login' : '/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(isLogin ? { account, password } : { account, password, display_name: displayName }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    renderAuthEntry(data.user);
    _setAuthStatus(`欢迎，${data.user.display_name || data.user.account}`, true);
    toast(isLogin ? '已登录' : '注册成功，已自动登录');
    setTimeout(closeAuthModal, 700);
  } catch (e) {
    _setAuthStatus((e.message || e) + (isLogin ? '' : '（换个账号试试？）'));
  } finally {
    _authBusy = false;
    if (authSubmitBtn) authSubmitBtn.disabled = false;
  }
}

async function logout() {
  try {
    await fetch('/auth/logout', { method: 'POST' });
  } catch (e) { /* 网络失败也按已退出处理（Cookie 交给过期） */ }
  renderAuthEntry(null);
  toast('已退出登录');
}

// 注销账号（隐私政策"删除权"）：确认后删除账号与全部登录会话，不可恢复
async function deleteAccount() {
  const me = await fetch('/auth/me').then(r => r.json()).catch(() => ({}));
  if (!me.user) { renderAuthEntry(null); return; }
  const ok = window.confirm(
    `确定注销账号「${me.user.display_name || me.user.account}」？\n\n` +
    '将立即删除你的账号信息与全部登录会话，且不可恢复。\n' +
    '保存在你浏览器本地的对话记录不受影响（清除浏览器数据即可删除）。'
  );
  if (!ok) return;
  try {
    const resp = await fetch('/auth/account', { method: 'DELETE' });
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      throw new Error(d.detail || `HTTP ${resp.status}`);
    }
    renderAuthEntry(null);
    closeAuthModal();
    toast('账号已注销，相关数据已删除');
  } catch (e) {
    _setAuthStatus('注销失败：' + (e.message || e));
  }
}

if (document.getElementById('authDeleteBtn')) {
  document.getElementById('authDeleteBtn').addEventListener('click', deleteAccount);
}

// ---- 事件绑定 ----
if (authBtn) {
  authBtn.addEventListener('click', () => {
    if (authBtn.dataset.loggedIn === '1') logout();
    else openAuthModal();
  });
}
if (authCloseBtn) authCloseBtn.addEventListener('click', closeAuthModal);
if (authModal) authModal.addEventListener('click', (e) => { if (e.target === authModal) closeAuthModal(); });
if (authSubmitBtn) authSubmitBtn.addEventListener('click', submitAuth);
document.querySelectorAll('.auth-tab').forEach(t =>
  t.addEventListener('click', () => _switchAuthTab(t.dataset.authTab)));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && authModal && !authModal.hidden) closeAuthModal();
});
// 在密码框上按回车直接提交
['authLoginPassword', 'authRegPassword', 'authLoginAccount', 'authRegAccount'].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('keydown', (e) => { if (e.key === 'Enter') submitAuth(); });
});

// 启动时恢复会话态
refreshAuthState();
