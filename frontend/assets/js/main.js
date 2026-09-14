// ============================================================
// 初始化
// ============================================================
const restored = loadState();
// 跨标签页同步：其他标签删除会话时，本标签也把该会话从列表剔除并持久化，
// 防止本标签随后 saveState 把残留的旧会话覆盖回本地存储（导致刷新后"复活"）。
window.addEventListener('storage', (e) => {
  if (e.key === STORAGE_KEY_DELETED) {
    loadDeletedConvIds();
    state.conversations = state.conversations.filter((c) => !deletedConvIds.has(c.id));
    renderConversations();
  } else if (e.key === STORAGE_KEY) {
    // 另一标签页写入了会话状态：本标签用“已删名单”再过滤一遍，
    // 防止另一标签在其内存里仍含已删会话、覆盖写回本地存储后本标签读到旧数据而“复活”。
    loadDeletedConvIds();
    state.conversations = state.conversations.filter((c) => !deletedConvIds.has(c.id));
    renderConversations();
  }
});
if (restored && state.currentConversation) {
  // 静默恢复会话状态（列表/草稿），但仍停留在首页，进入对话时自然呈现
  queryInput.placeholder = '想与 TA 谈些什么…';
  updateEmptyState();
  renderConversations();
  renderMessages();
  queryInput.value = state.currentConversation._draft || '';
  autoResize(queryInput);
  console.log('[App] 从 localStorage 恢复会话(后台), Session:', state.sessionId);
}
// 永远先进入首页介绍页，选定人物后再进入对话
state.view = 'home';
homeView.classList.remove('hidden');
homeStage('intro');
console.log('[App] 首页就绪, 等待开始');

fetch('/health').then(r => r.json()).then(d => {
  console.log('[Health]', d.message);
  setStatus('idle', '就绪');
}).catch(() => setStatus('busy', '服务未连接'));

applyTheme(themeFor(state.currentCharacter));
loadCharacters();
