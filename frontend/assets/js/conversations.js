// ============================================================
// 空状态 & 角色
// ============================================================
function updateEmptyState() {
  const char = currentChar();
  if (heroAvatar) heroAvatar.textContent = nameMark(char?.name);
  if (heroMono) heroMono.textContent = nameMark(char?.name);
  if (heroName) heroName.textContent = char?.name || '对话';
  if (heroDesc) heroDesc.textContent = char?.description || '';
  if (heroGreet) heroGreet.textContent = char?.tagline || '与历史伟人对谈，在对话中照见自己。';
  if (topAvatar) topAvatar.textContent = nameMark(char?.name);
  if (topName) topName.textContent = char?.name || '';
  if (topDesc) topDesc.textContent = char?.description || '';
  if (charHint) charHint.textContent = `当前与「${char?.name || ''}」对话`;
  renderSuggestions();
}

function renderSuggestions() {
  const charId = state.currentCharacter;
  const list = SUGGESTIONS[charId] || DEFAULT_SUGGESTIONS;
  suggestionChips.innerHTML = '';
  list.forEach(q => {
    const chip = document.createElement('button');
    chip.className = 'suggestion-chip';
    chip.textContent = q;
    chip.addEventListener('click', () => { if (!state.isLoading) sendQuery(q); });
    suggestionChips.appendChild(chip);
  });
}

function updateEmptyLayout() {
  const isPersonaEmpty = !state.currentConversation || state.currentConversation.messages.length === 0;
  document.querySelector('.main-area').classList.toggle('centered-empty', isPersonaEmpty);
  // 空对话时中间已展示人物卡片，左上角不再重复显示人物（有消息后恢复显示）
  if (topbarPersona) topbarPersona.style.display = isPersonaEmpty ? 'none' : '';
}

function makeZoneLabel(text) {
  const el = document.createElement('div');
  el.className = 'char-zone-label';
  el.textContent = text;
  return el;
}

function makeCharCard(char) {
  const card = document.createElement('div');
  card.className = 'char-card' + (char.id === state.currentCharacter ? ' active' : '');
  card.dataset.charId = char.id;
  const delBtn = char.is_custom
    ? `<button class="char-del" data-del-id="${escapeHtml(char.id)}" title="删除自建角色">×</button>`
    : '';
  card.innerHTML = `
    <span class="char-avatar">${escapeHtml(nameMark(char.name))}</span>
    <div>
      <div class="char-name">${escapeHtml(char.name)}</div>
      <div class="char-desc">${escapeHtml(char.description || '')}</div>
    </div>${delBtn}`;
  card.addEventListener('click', () => switchCharacter(char.id));
  if (char.is_custom) {
    const db = card.querySelector('.char-del');
    if (db) db.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteCustomCharacter(char.id, char.name);
    });
  }
  return card;
}

async function loadCharacters() {
  try {
    const resp = await fetch('/persona/characters');
    if (!resp.ok) return;
    const data = await resp.json();
    state.characters = data.characters || [];

    charCards.innerHTML = '';
    const eduChars = state.characters.filter(c => (c.zone || 'education') === 'education');
    const entChars = state.characters.filter(c => (c.zone || 'education') === 'entertainment');
    if (eduChars.length) {
      charCards.appendChild(makeZoneLabel('问道'));
      eduChars.forEach(c => charCards.appendChild(makeCharCard(c)));
    }
    if (entChars.length) {
      charCards.appendChild(makeZoneLabel('会心'));
      entChars.forEach(c => charCards.appendChild(makeCharCard(c)));
    }

    updateEmptyState();
    updateEmptyLayout();
    // 角色数据就绪后刷新侧边栏：历史对话头像/名字 + 当前对话角色高亮
    if (state.conversations.length > 0) {
      const cid = getConversationCharacter(state.currentConversation);
      if (cid && cid !== state.currentCharacter) {
        state.currentCharacter = cid;
        applyTheme(themeFor(cid));
        updateEmptyState();
        charCards.querySelectorAll('.char-card').forEach(c =>
          c.classList.toggle('active', c.dataset.charId === cid)
        );
      }
      renderConversations();
      // 页面恢复时也用后端权威记录核对当前对话归属
      verifyConversationOwner(state.currentConversation);
    }
  } catch (e) {
    console.warn('[App] 加载角色列表失败:', e);
  }
  if (state.view === 'home') renderHome();
}

function switchCharacter(charId) {
  if (charId === state.currentCharacter) return;
  state.currentCharacter = charId;
  applyTheme(themeFor(charId));
  state.sessionId = crypto.randomUUID();
  updateEmptyState();
  newConversation();
  charCards.querySelectorAll('.char-card').forEach(c =>
    c.classList.toggle('active', c.dataset.charId === charId)
  );
  closeSidebar();
}

// ============================================================
// 对话管理
// ============================================================
function newConversation() {
  if (state.currentConversation) {
    state.currentConversation._draft = queryInput.value;
  }
  if (state.currentConversation
      && state.currentConversation.messages.length === 0
      && !state.currentConversation._fromHistory
      && (!state.currentConversation.scene || state.currentConversation.scene === state.currentScene)) {
    state.sessionId = crypto.randomUUID();
    state.currentConversation.sessionId = state.sessionId;
    state.currentConversation.title = '新对话';
    state.currentConversation.characterId = state.currentCharacter;
    saveState();
    renderConversations();
    renderMessages();
    queryInput.value = state.currentConversation._draft || '';
    queryInput.focus();
    emptyState.style.display = 'flex';
    return;
  }

  // 查找其他可复用的空对话：避免重复创建空对话（如点开历史对话后再切换角色）
  const reusable = state.conversations.find(c =>
      (!c.messages || c.messages.length === 0)
      && (!c.scene || c.scene === state.currentScene));
  if (reusable) {
    state.currentConversation = reusable;
    delete reusable._fromHistory;
    state.sessionId = crypto.randomUUID();
    reusable.sessionId = state.sessionId;
    reusable.title = '新对话';
    reusable.characterId = state.currentCharacter;
    saveState();
    renderConversations();
    renderMessages();
    queryInput.value = reusable._draft || '';
    queryInput.focus();
    emptyState.style.display = 'flex';
    return;
  }

  state.sessionId = crypto.randomUUID();
  state.currentConversation = {
    id: crypto.randomUUID(),
    title: '新对话',
    messages: [],
    createdAt: Date.now(),
    sessionId: state.sessionId,
    scene: state.currentScene,
    characterId: state.currentCharacter,
  };
  state.conversations.push(state.currentConversation);
  saveState();
  renderConversations();
  renderMessages();
  queryInput.value = '';
  queryInput.focus();
  emptyState.style.display = 'flex';
}

function addMessage(type, content, extra = {}) {
  if (!state.currentConversation) newConversation();
  if (!state.currentConversation.sessionId) {
    state.currentConversation.sessionId = state.sessionId;
  }
  state.currentConversation.messages.push({ type, content, extra, time: Date.now() });
  if (type === 'user' && state.currentConversation.messages.length === 1) {
    state.currentConversation.title = content.slice(0, 30) + (content.length > 30 ? '…' : '');
  }
  if (type !== 'assistant') {
    state._animateFromIndex = state.currentConversation.messages.length - 1;
  }
  renderConversations();
  renderMessages();
  saveState();
}

function renderConversations() {
  const sceneConvs = currentCharacterConvs();
  if (!sceneConvs || sceneConvs.length === 0) {
    conversationList.innerHTML = '<div class="conv-empty">— 暂无历史对话 —</div>';
    return;
  }
  let html = '';
  for (let i = sceneConvs.length - 1; i >= 0; i--) {
    const conv = sceneConvs[i];
    const isActive = conv.id === state.currentConversation?.id;
    const cid = getConversationCharacter(conv);
    const cdef = state.characters.find(c => c.id === cid) || null;
    const avatar = nameMark(cdef?.name);
    const who = cdef ? cdef.name : (cid ? cid : '');
    html += `<div class="conv-item${isActive ? ' active' : ''}" data-conv-id="${conv.id}" title="${escapeHtml(conv.title)}">
      <span class="conv-avatar">${avatar}</span>
      <span class="conv-body">
        <span class="conv-title">${escapeHtml(conv.title)}</span>
        <span class="conv-sub"><span class="conv-who">${escapeHtml(who)}</span><span class="conv-time">${formatConvTime(conv.createdAt || Date.now())}</span></span>
      </span>
      <button class="conv-del" data-delete-id="${conv.id}" title="删除对话">×</button>
    </div>`;
  }
  conversationList.innerHTML = html;

  conversationList.querySelectorAll('.conv-item').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.classList.contains('conv-del')) return;
      const convId = el.dataset.convId;
      const target = state.conversations.find(c => c.id === convId);
      if (target && target !== state.currentConversation) {
        if (state.currentConversation) state.currentConversation._draft = queryInput.value;
        // 标记为从历史列表点开的对话：切换角色时不可被空对话复用逻辑改写归属
        target._fromHistory = true;
        state.currentConversation = target;
        state.sessionId = target.sessionId || crypto.randomUUID();
        state._animateFromIndex = 0;
        // 同步对话对应的角色：该角色的头像常亮（点击历史对话后总是同步）
        const cid = getConversationCharacter(target);
           if (cid) {
             state.currentCharacter = cid;
             applyTheme(themeFor(cid));
             updateEmptyState();
          charCards.querySelectorAll('.char-card').forEach(c =>
            c.classList.toggle('active', c.dataset.charId === cid)
          );
        }
        // 后端权威核对：历史对话的对象优先级最高，后端有记录则强制修正归属
        verifyConversationOwner(target);
        saveState();
        renderConversations();
        renderMessages();
        queryInput.value = target._draft || '';
        autoResize(queryInput);
        closeSidebar();
      }
    });
  });

  conversationList.querySelectorAll('.conv-del').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(btn.dataset.deleteId);
    });
  });
}

function deleteConversation(convId) {
  const idx = state.conversations.findIndex(c => c.id === convId);
  if (idx === -1) return;
  const deletedConv = state.conversations[idx];
  const wasCurrent = deletedConv.id === state.currentConversation?.id;
  state.conversations.splice(idx, 1);
  deletedConvIds.add(convId);
  saveDeletedConvIds();

  if (wasCurrent) {
    // 删除当前对话后跳到一个全新空会话，而不是自动选中另一个可能带旧历史的会话，
    // 否则用户直接输入同样问题时 AI 会带着那个会话的旧上下文，看起来像"复活"。
    newConversation();
  }

  saveState();
  renderConversations();
  renderMessages();
  queryInput.value = state.currentConversation?._draft || '';
  autoResize(queryInput);

  if (deletedConv.sessionId) {
    fetch('/conversation/' + encodeURIComponent(deletedConv.sessionId), {
      method: 'DELETE',
      keepalive: true,
    }).catch(() => {});
  }
}
