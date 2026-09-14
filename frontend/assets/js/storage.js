// ============================================================
// 持久化（localStorage + sessionStorage 双写）
// ============================================================
const STORAGE_KEY = 'multi_agent_rag_state';

// 已删除会话 id 名单（持久化兜底）：删除按钮只需改内存 + 本地某个标签的 state，
// 但极端情况下（多标签页时另一标签把含旧会话的状态覆盖写回本地存储、或 saveState
// 竞态）被删会话会残留在持久化 JSON 里，导致刷新/重启后"复活"。
// 这里额外记一份"已删名单"，加载时强制剔除，确保被删对话永不再出现。
const STORAGE_KEY_DELETED = 'multi_agent_rag_deleted';
const deletedConvIds = new Set();
function loadDeletedConvIds() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_DELETED);
    if (raw) JSON.parse(raw).forEach((id) => deletedConvIds.add(id));
  } catch (e) { /* 忽略 */ }
}
function saveDeletedConvIds() {
  try {
    localStorage.setItem(STORAGE_KEY_DELETED, JSON.stringify([...deletedConvIds]));
  } catch (e) { /* 忽略 */ }
}
loadDeletedConvIds();

function saveState() {
  try {
    const data = {
      conversations: state.conversations,
      currentConversationId: state.currentConversation?.id || null,
      currentScene: state.currentScene,
      currentCharacter: state.currentCharacter,
      sessionId: state.sessionId,
    };
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch (e) { /* 忽略存储配额超限 */ }
}

function loadState() {
  try {
    let raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    if (data.conversations && data.conversations.length > 0) {
      state.conversations = (data.conversations || [])
        .filter(c => !c.scene || c.scene === 'persona')
        .filter(c => !deletedConvIds.has(c.id))
        .map(c => { delete c._fromHistory; return c; });
      state.currentScene = 'persona';
      state.currentCharacter = data.currentCharacter || 'jung';
      state.sessionId = data.sessionId || crypto.randomUUID();
      if (data.currentConversationId) {
        state.currentConversation = state.conversations.find(
          c => c.id === data.currentConversationId && (!c.scene || c.scene === state.currentScene)
        ) || null;
      }
      if (!state.currentConversation) {
        const sceneConvs = state.conversations.filter(c => !c.scene || c.scene === state.currentScene);
        state.currentConversation = sceneConvs.length > 0 ? sceneConvs[sceneConvs.length - 1] : null;
      }
      if (state.currentConversation) {
        if (state.currentConversation.sessionId) {
          state.sessionId = state.currentConversation.sessionId;
        } else {
          state.currentConversation.sessionId = state.sessionId || crypto.randomUUID();
          state.sessionId = state.currentConversation.sessionId;
        }
        // 恢复对话对应的角色（按对话实际归属判断，保持与侧边栏一致）
        const restoredCid = getConversationCharacter(state.currentConversation);
        if (restoredCid) {
          state.currentCharacter = restoredCid;
        }
      }
      return true;
    }
  } catch (e) { console.warn('[持久化] 恢复失败:', e); }
  return false;
}
