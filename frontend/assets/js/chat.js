// ============================================================
// 消息渲染
// ============================================================
// 处理进度文案：节点名是实现细节（retrieval_agent 这种），不该直接甩给用户看。
// 统一映射成"正在做什么"，新增节点记得在这里补一条。
const AGENT_STEP_LABELS = {
  supervisor: '判断该怎么回应',
  tool_agent: '斟酌要不要查资料',
  retrieval_agent: '翻查原著',
  analysis_agent: '组织回答',
  verifier: '核对引用',
};

// 酒馆式开场白气泡：角色先开口，进对话即见。
// 这是角色卡里写死的一句话，不走 LLM、不入库、不进入对话历史，纯粹消除"空白页"的出戏感。
function openingBubbleHTML(text) {
  const icon = getAssistantIcon();
  return `
    <div class="message-group msg-animate">
      <div class="message-row">
        <div class="avatar assistant">${icon}</div>
        <div class="message-content">
          <div class="md-body">${renderMarkdownLite(text)}</div>
          <div class="message-ts">${formatTime()}</div>
        </div>
      </div>
    </div>`;
}

function renderMessages() {
  emptyState.style.display = 'none';
  const msgs = state.currentConversation?.messages || [];
  updateEmptyLayout();
  if (msgs.length === 0) {
    emptyState.style.display = 'flex';
    chatScroll.innerHTML = '';
    const opening = currentChar()?.first_mes || '';
    if (opening) chatScroll.innerHTML = openingBubbleHTML(opening);
    chatScroll.appendChild(emptyState);
    return;
  }

  // 定位最后一条助手消息：操作条（重答/朗读）只出现在最新回答上
  let lastAssistantIdx = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].type === 'assistant') { lastAssistantIdx = i; break; }
  }

  let html = '';
  for (let i = 0; i < msgs.length; i++) {
    const msg = msgs[i];
    const animClass = (i >= state._animateFromIndex) ? ' msg-animate' : '';

    if (msg.type === 'user') {
      html += `
        <div class="message-group${animClass}">
          <div class="message-row user-row">
            <div class="avatar user">我</div>
            <div class="message-content">
              <div class="user-text">${escapeHtml(msg.content)}<button class="user-edit-btn" type="button" data-edit-idx="${i}" title="编辑这条消息（之后的对话将作废）">✎ 编辑</button></div>
              <div class="message-ts user-ts">${formatTime()}</div>
            </div>
          </div>
        </div>`;
    } else if (msg.type === 'assistant') {
      const agents = msg.extra?.agents || [];
      const badges = agents.length ? agents.map(a =>
        `<span class="agent-badge">${escapeHtml(String(a).replace(/_agent$/, ''))}</span>`
      ).join('') : '';
      const icon = getAssistantIcon();
      const graphTag = msg.extra?.graph_used
        ? '<span class="agent-badge kg-badge" title="本轮回答由 AI 自主调用知识图谱增强生成">已调用知识图谱</span>' : '';
      const recoveredTag = msg.extra?.recovered
        ? '<span class="recovered-tag">刷新后恢复</span>' : '';
      // 引用核查报告：元信息，默认折叠。仅当正文真的引用了资料（含 [n] 角标）
      // 时才出现——寒暄类回答正文不含角标，因此不会再看到那张引用可信度评分表。
      const citeReport = (msg.extra?.citations && /\[\d{1,2}\]/.test(msg.content || ''))
        ? `<details class="cite-report"><summary>引用出处</summary><div class="cite-report-body">${escapeHtml(msg.extra.citations)}</div></details>`
        : '';
      // 多版本回答（重新生成产生的候选）：可在版本间来回切换
      const variants = (msg.extra?.variants && msg.extra.variants.length > 1) ? msg.extra.variants : null;
      const vIdx = variants ? Math.min(msg.extra.variantIndex ?? variants.length - 1, variants.length - 1) : 0;
      const variantBar = variants ? `
              <div class="variant-switch">
                <button class="msg-act" type="button" data-variant-idx="${i}" data-variant-dir="-1" ${vIdx <= 0 ? 'disabled' : ''}>‹</button>
                <span>第 ${vIdx + 1}/${variants.length} 版</span>
                <button class="msg-act" type="button" data-variant-idx="${i}" data-variant-dir="1" ${vIdx >= variants.length - 1 ? 'disabled' : ''}>›</button>
              </div>` : '';
      const actions = (i === lastAssistantIdx && !state.isLoading) ? `
              <div class="msg-actions">
                <button class="msg-act" type="button" data-regen="1" title="换个说法，重新回答这条消息">↻ 重新生成</button>
                <button class="msg-act${state._ttsIdx === i ? ' tts-on' : ''}" type="button" data-tts-idx="${i}" title="朗读这段回答">${state._ttsIdx === i ? '停止' : '朗读'}</button>
              </div>` : '';
      html += `
        <div class="message-group${animClass}">
          <div class="message-row">
            <div class="avatar assistant">${icon}</div>
            <div class="message-content">
              <div class="md-body">${renderMarkdownLite(isImmersiveTheme() ? stripMetaSections(msg.content) : msg.content, msg.extra?.doc_map)}</div>
              ${citeReport}
              ${(badges || graphTag) ? `<div class="agent-badges">${badges}${graphTag}</div>` : ''}${recoveredTag}
              ${actions}${variantBar}
              <div class="message-ts"><span class="ai-tag" title="本条内容由人工智能生成（AI-generated content）">AI 生成</span>${formatTime()}</div>
            </div>
          </div>
        </div>`;
    } else if (msg.type === 'system') {
      html += `
        <div class="message-group${animClass}">
          <div class="message-row center-row">
            <span class="sys-chip">${escapeHtml(msg.content)}</span>
          </div>
        </div>`;
    } else if (msg.type === 'error') {
      const retryBtn = msg.extra?.retryQuery
        ? `<button class="err-retry" type="button" data-retry-query="${escapeHtml(msg.extra.retryQuery)}">重试</button>`
        : '';
      html += `
        <div class="message-group${animClass}">
          <div class="message-row center-row">
            <span class="err-chip">${escapeHtml(msg.content)}</span>${retryBtn}
          </div>
        </div>`;
    }
  }

  state._animateFromIndex = msgs.length;

  if (state.isLoading) {
    const icon = getAssistantIcon();
    html += `
      <div class="message-group" id="thinkingIndicator">
        <div class="message-row">
          <div class="avatar assistant">${icon}</div>
          <div class="message-content">
            <div class="thinking-indicator">
              <span>正在生成回答</span>
              <div class="thinking-dots"><span></span><span></span><span></span></div>
            </div>
            <div class="thinking-sub" id="thinkingSubtext"></div>
          </div>
        </div>
      </div>`;
  }

  chatScroll.innerHTML = html;
  // 失败消息附带"重试"：一键重发刚才的问题（先移除失败轮次，避免消息重复）
  chatScroll.querySelectorAll('.err-retry').forEach(btn => {
    btn.addEventListener('click', () => {
      if (state.isLoading) return;
      const q = btn.dataset.retryQuery;
      if (!q || !state.currentConversation) return;
      const msgs = state.currentConversation.messages;
      for (let i = msgs.length - 1; i >= 0; i--) {
        if (msgs[i].type === 'error' && msgs[i].extra?.retryQuery === q) {
          if (i > 0 && msgs[i - 1].type === 'user' && msgs[i - 1].content === q) msgs.splice(i - 1, 2);
          else msgs.splice(i, 1);
          break;
        }
      }
      sendQuery(q);
    });
  });
  // ---- 新增操作绑定：历史编辑 / 重新生成 / 朗读 / 版本切换 ----
  chatScroll.querySelectorAll('.user-edit-btn').forEach(btn => {
    btn.addEventListener('click', () => editUserMessage(Number(btn.dataset.editIdx)));
  });
  chatScroll.querySelectorAll('[data-regen]').forEach(btn => {
    btn.addEventListener('click', regenerateLast);
  });
  chatScroll.querySelectorAll('[data-tts-idx]').forEach(btn => {
    btn.addEventListener('click', () => ttsToggle(Number(btn.dataset.ttsIdx)));
  });
  chatScroll.querySelectorAll('[data-variant-idx]').forEach(btn => {
    btn.addEventListener('click', () =>
      switchVariant(Number(btn.dataset.variantIdx), Number(btn.dataset.variantDir)));
  });
  scrollToBottom();
}

// ============================================================
// SSE 查询
// ============================================================
async function sendQuery(query, opts = {}) {
  if (state.isLoading || !query.trim()) return;

  // 对话归属当前角色，历史列表据此显示头像
  if (state.currentConversation) {
    state.currentConversation.characterId = state.currentCharacter;
  }

  // 发送前统计完整问答对数：随请求告诉后端"用户此刻看到的对话就到第几轮"，
  // 后端把存储的历史截断到同一位置（编辑历史消息后两端才能保持一致；
  // 正常发送时该值与后端现状一致，截断是空操作，不会丢轮次）
  let completePairs = 0;
  const msgsBefore = state.currentConversation?.messages || [];
  for (let i = 0; i < msgsBefore.length; i++) {
    if (msgsBefore[i].type === 'user' && msgsBefore[i + 1]?.type === 'assistant') {
      completePairs++; i++;
    }
  }

  addMessage('user', query);
  queryInput.value = '';
  autoResize(queryInput);
  sendBtn.disabled = true;
  state.isLoading = true;
  state._streamingText = '';
  state._isStreaming = true;
  state._gotToken = false;
  state._reqStart = Date.now();
  setStatus('busy', '思考中');

  const waitTimer = setInterval(() => {
    if (state._gotToken || !state.isLoading) return;
    const secs = Math.max(1, Math.round((Date.now() - state._reqStart) / 1000));
    updateThinkingText(`正在检索资料与思考中（已等待 ${secs} 秒，请勿刷新页面）…`);
  }, 15000);
  renderMessages();

  try {
    const body = { query, session_id: state.sessionId, character_id: state.currentCharacter };
    body.truncate_to_turns = completePairs;
    if (opts.regenerate) body.regenerate = true;
    const msgs = state.currentConversation?.messages || [];
    const pairedHistory = [];
    for (let i = 0; i < msgs.length; i++) {
      const m = msgs[i];
      if (m.type !== 'user' && m.type !== 'assistant') continue;
      if (m.type === 'user') {
        const next = msgs[i + 1];
        if (next && next.type === 'assistant') {
          pairedHistory.push({ type: 'user', content: m.content });
          pairedHistory.push({ type: 'assistant', content: next.content });
          i++;
        }
      } else {
        pairedHistory.push({ type: 'assistant', content: m.content });
      }
    }
    body.history = pairedHistory;

    const resp = await fetch('/persona/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullContent = '';
    let routeHistory = [];
    let graphUsedThisTurn = false;
    let docMapThisTurn = null;
    // 核查报告单独承载：它是元信息，不能并入正文（并入后会被当成角色说的话）
    let citationsThisTurn = '';
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
        switch (data.type) {
          case 'thinking': updateThinkingText(data.content); break;
          case 'token':
            state._gotToken = true;
            state._streamingText += data.content || '';
            updateStreamingBubble();
            break;
          case 'agent_done': updateThinkingText(`正在${AGENT_STEP_LABELS[data.agent] || '处理'}…`); break;
          case 'trace': routeHistory = data.route_history || []; break;
          case 'result': fullContent = data.content || ''; graphUsedThisTurn = !!data.graph_used; docMapThisTurn = data.doc_map || null; break;
          case 'citations':
            // 核查报告不并入正文：并入后它会成为"角色说的话"的一部分，
            // 连寒暄（"你来了。秋夜山深…"）后面都会跟着一张引用可信度评分表。
            // 改由 extra.citations 单独承载，渲染时默认折叠，且仅当正文真的
            // 引用了资料（含 [n] 角标）时才展示。
            if (data.content) {
              citationsThisTurn = data.content;
              graphUsedThisTurn = !!data.graph_used || graphUsedThisTurn;
            }
            break;
          case 'questions':
            fullContent = data.content || '';
            state._pendingQuestions = data.questions || [];
            graphUsedThisTurn = !!data.graph_used;
            docMapThisTurn = data.doc_map || docMapThisTurn;
            break;
          case 'error': throw new Error(data.content);
        }
      }
    }

    if (buffer.trim().startsWith('data: ')) {
      try {
        const data = JSON.parse(buffer.trim().slice(6));
        if (data.type === 'result') { fullContent = data.content || ''; graphUsedThisTurn = !!data.graph_used; docMapThisTurn = data.doc_map || docMapThisTurn; }
        if (data.type === 'questions') { fullContent = data.content || ''; state._pendingQuestions = data.questions || []; graphUsedThisTurn = !!data.graph_used; docMapThisTurn = data.doc_map || docMapThisTurn; }
      } catch (e) { /* 忽略 */ }
    }

    state._isStreaming = false;
    state._streamingText = '';
    state.isLoading = false;
    clearInterval(waitTimer);

    if (fullContent) {
      const msgData = {
        type: 'assistant',
        content: fullContent,
        extra: { agents: routeHistory, sceneIcon: getAssistantIcon(), graph_used: graphUsedThisTurn, doc_map: docMapThisTurn, citations: citationsThisTurn || null },
      };
      // 重新生成：把历史版本带上，前端可在各版本之间切换回看
      if (state._regenVariants && state._regenVariants.length) {
        msgData.extra.variants = [...state._regenVariants, fullContent];
        msgData.extra.variantIndex = msgData.extra.variants.length - 1;
        state._regenVariants = null;
      }
      state._msgQueue = [msgData];
      graphUsedThisTurn = false;
      docMapThisTurn = null;
      citationsThisTurn = '';
      await processMessageQueue();
    } else {
      addMessage('system', '查询完成，但未返回内容。');
      renderMessages();
    }

  } catch (err) {
    state._isStreaming = false;
    state._streamingText = '';
    state.isLoading = false;
    clearInterval(waitTimer);
    addMessage('error', `查询失败：${err.message}，可点击"重试"再次发送。`, { retryQuery: query });
  }

  state.isLoading = false;
  renderMessages();
  const leftover = document.getElementById('thinkingIndicator');
  if (leftover) leftover.remove();
  const leftoverBubble = document.getElementById('streamingBubble');
  if (leftoverBubble) leftoverBubble.remove();
  sendBtn.disabled = !queryInput.value.trim();
  autoResize(queryInput);
  setStatus('idle', '就绪');
}

function setStatus(mode, text) {
  statusDot.classList.toggle('busy', mode === 'busy');
  statusText.textContent = text;
}

// ============================================================
// 对话操作：重新生成 / 编辑历史 / 版本切换 / 朗读
// ============================================================

// 重新生成最后一条回答：移除最后一轮问答，带着历史版本重发同一问题。
// 后端会作废旧答案（弹出最后一轮 + 跳过答案缓存），给出一个新的回答；
// 旧回答保留在消息的 variants 里，可用 ‹ › 随时切回。
async function regenerateLast() {
  if (state.isLoading || !state.currentConversation) return;
  const msgs = state.currentConversation.messages;
  let ai = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].type === 'assistant') { ai = i; break; }
  }
  if (ai <= 0 || msgs[ai - 1].type !== 'user') return;

  const prevVariants = (msgs[ai].extra?.variants && msgs[ai].extra.variants.length)
    ? msgs[ai].extra.variants.slice() : [msgs[ai].content];
  const q = msgs[ai - 1].content;

  // 正在朗读的消息若被移除，先停止朗读
  if (state._ttsIdx >= ai - 1) { stopTTS(); }

  msgs.splice(ai - 1, 2);
  state._animateFromIndex = Math.min(msgs.length, ai - 1);
  state._regenVariants = prevVariants;
  saveState();
  renderMessages();
  try {
    await sendQuery(q, { regenerate: true });
  } finally {
    state._regenVariants = null;
  }
}

// 编辑某条历史消息：该消息及其之后的对话作废，内容放回输入框修改后重发。
// 重发时 sendQuery 会自动携带 truncate_to_turns，后端同步截断存储的历史。
function editUserMessage(idx) {
  if (state.isLoading || !state.currentConversation) return;
  const msgs = state.currentConversation.messages;
  const msg = msgs[idx];
  if (!msg || msg.type !== 'user') return;
  const text = msg.content;
  if (state._ttsIdx >= idx) { stopTTS(); }
  msgs.splice(idx);
  saveState();
  renderConversations();
  renderMessages();
  queryInput.value = text;
  autoResize(queryInput);
  sendBtn.disabled = false;
  queryInput.focus();
}

// 多版本回答切换：把指定版本设为当前内容（历史配对随 msg.content 走）
function switchVariant(idx, dir) {
  if (state.isLoading || !state.currentConversation) return;
  const msg = state.currentConversation.messages[idx];
  const variants = msg?.extra?.variants;
  if (!variants || variants.length < 2) return;
  const cur = msg.extra.variantIndex ?? variants.length - 1;
  const next = cur + dir;
  if (next < 0 || next >= variants.length) return;
  msg.extra.variantIndex = next;
  msg.content = variants[next];
  saveState();
  renderMessages();
}

// ---- 朗读（浏览器内置语音合成，零后端成本）----
function cleanTextForTTS(text) {
  return (text || '')
    .replace(/```[\s\S]*?```/g, '（代码略）')
    .replace(/\n---[\s\S]*$/, '')          // 引用出处区块不读
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/[#>`|]/g, '')
    .replace(/\s*\n\s*/g, ' ')
    .trim();
}

function stopTTS() {
  try { window.speechSynthesis?.cancel(); } catch (e) { /* 忽略 */ }
  state._ttsIdx = -1;
}

function ttsToggle(idx) {
  const synth = window.speechSynthesis;
  if (!synth) return;
  if (state._ttsIdx === idx) { stopTTS(); renderMessages(); return; }
  synth.cancel();
  const msg = state.currentConversation?.messages?.[idx];
  const text = cleanTextForTTS(msg?.content);
  if (!text) return;
  const u = new SpeechSynthesisUtterance(text.slice(0, 1200));
  u.lang = 'zh-CN';
  u.rate = 1.05;
  u.onend = () => {
    if (state._ttsIdx === idx) { state._ttsIdx = -1; renderMessages(); }
  };
  u.onerror = () => {
    if (state._ttsIdx === idx) { state._ttsIdx = -1; renderMessages(); }
  };
  state._ttsIdx = idx;
  renderMessages();
  synth.speak(u);
}

function updateThinkingText(text) {
  const el = document.getElementById('thinkingSubtext');
  if (el) el.textContent = text;
}

function updateStreamingBubble() {
  let bubble = document.getElementById('streamingBubble');
  if (!bubble) {
    const thinking = document.getElementById('thinkingIndicator');
    if (thinking) {
      thinking.outerHTML = `
        <div class="message-group msg-animate" id="streamingBubble">
          <div class="message-row">
            <div class="avatar assistant">${getAssistantIcon()}</div>
            <div class="message-content">
              <div class="md-body streaming-cursor" id="streamingContent"></div>
            </div>
          </div>
        </div>`;
    }
    bubble = document.getElementById('streamingBubble');
  }
  if (bubble) {
    const contentEl = document.getElementById('streamingContent');
    if (contentEl) contentEl.innerHTML = renderMarkdownLite(isImmersiveTheme() ? stripMetaSections(state._streamingText) : state._streamingText);
    scrollToBottom();
  }
}

// ============================================================
// 轻量 Markdown 渲染（流式 + 最终消息通用）
// ============================================================
function isImmersiveTheme() {
  const t = themeFor(state.currentCharacter) || document.body.dataset.theme;
  return t === 'street' || t === 'live';
}

// 沉浸主题：剥离「要点/关键点/来源/引用」等出戏段落
function stripMetaSections(text) {
  if (!text) return text;
  // 仅剔除明确的系统内部标记行（模型不会自然产出这类内容），
  // 绝不删除人物正常回答中的"要点 / 来源 / 引用 / 参考"等段落，避免内容丢失。
  return text.split('\n').filter(line => {
    const t = line.trim();
    return !/^【系统提示】|^【内部】|^（内部：/.test(t);
  }).join('\n');
}

function renderMarkdownLite(text, docMap) {
  if (!text) return '';
  let html = escapeHtml(text);
  // 内联引用角标：正文 [n] → 悬浮显示来源著作与章节（docMap 由后端 result 事件携带，
  // 编号与提示词里的资料清单一一对应；verifier 的"#### N."编号格式不匹配本规则，不受影响）
  if (Array.isArray(docMap) && docMap.length) {
    html = html.replace(/\[(\d{1,2})\]/g, (m, n) => {
      const d = docMap.find(x => +x.n === +n);
      if (!d) return m;
      const tip = [d.source, d.heading].filter(Boolean).join(' · ');
      return `<sup class="cite-mark" title="${escapeHtml(tip)}">[${n}]</sup>`;
    });
  }
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/(?:^|&lt;br&gt;)\s*\*([^*&lt;]+)\*/gm, '<span class="action-text">$1</span>');
  html = html.replace(/(?:^|&lt;br&gt;)\s*(\u3010[^\u3011]+\u3011)/g, '<span class="action-text">$1</span>');
  html = html.replace(/(?:^|&lt;br&gt;)\s*(\uff08[^\uff09]+\uff09)/g, '<span class="action-text">$1</span>');
  html = html.replace(/(?:^|&lt;br&gt;)\s*(\([^)]+\))/g, '<span class="action-text">$1</span>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');
  html = '<p>' + html + '</p>';
  html = html.replace(/<p>\s*<\/p>/g, '');
  return html;
}

// ============================================================
// 消息队列
// ============================================================
async function processMessageQueue() {
  const streamingBubble = document.getElementById('streamingBubble');
  if (streamingBubble) streamingBubble.remove();

  while (state._msgQueue.length > 0) {
    const msgData = state._msgQueue.shift();
    if (!state.currentConversation) newConversation();
    state.currentConversation.messages.push({ ...msgData, time: Date.now() });
    renderMessages();
    if (state._msgQueue.length > 0) {
      await new Promise(r => setTimeout(r, 700));
    }
  }
  saveState();
}

// ============================================================
// 刷新后恢复未收到的回答
// ============================================================
async function recoverPendingAnswer() {
  const conv = state.currentConversation;
  if (!conv || !conv.sessionId) return;
  const msgs = conv.messages || [];
  if (msgs.length === 0) return;
  const last = msgs[msgs.length - 1];
  if (last.type !== 'user') return;

  const pendingQuery = last.content;
  const systemMsg = {
    type: 'system',
    content: '检测到上一轮回答未收到，正在从后端恢复…',
    extra: {},
    time: Date.now(),
  };
  msgs.push(systemMsg);
  saveState();
  renderMessages();

  const url = '/conversation/pending/' + encodeURIComponent(conv.sessionId);
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    if (state.currentConversation !== conv) return;
    if (!state.conversations.includes(conv)) return;
    if (msgs[msgs.length - 1] !== systemMsg) return;
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        if (data.final_answer && data.query === pendingQuery) {
          if (state.currentConversation !== conv) return;
          const idx = msgs.indexOf(systemMsg);
          if (idx === -1) return;
          msgs.splice(idx, 1, {
            type: 'assistant',
            content: data.final_answer,
            extra: {
              agents: data.route_history || [],
              sceneIcon: getAssistantIcon(),
              recovered: true,
            },
            time: Date.now(),
          });
          if (data.info_gap_questions && data.info_gap_questions.length > 0) {
            state._pendingQuestions = data.info_gap_questions;
          }
          saveState();
          renderMessages();
          scrollToBottom();
          return;
        }
        if (data.final_answer && data.query !== pendingQuery) {
          systemMsg.content = '上一轮回答未能恢复（与当前问题不匹配），请重新提问。';
          saveState();
          renderMessages();
          return;
        }
      }
    } catch (e) { /* 网络瞬时错误，继续轮询 */ }
    await new Promise(r => setTimeout(r, 2000));
  }
  systemMsg.content = '等待超时，上一轮回答未能恢复，请重新提问。';
  saveState();
  renderMessages();
}

// ============================================================
// 事件绑定
// ============================================================
// ============ 使用须知弹层 ============
const noticeModal = document.getElementById('noticeModal');
function openNotice() { noticeModal.hidden = false; }
function closeNotice() { noticeModal.hidden = true; }
document.getElementById('homeNoticeBtn').addEventListener('click', openNotice);
document.getElementById('sideNoticeBtn').addEventListener('click', openNotice);
document.getElementById('noticeCloseBtn').addEventListener('click', closeNotice);
noticeModal.addEventListener('click', (e) => { if (e.target === noticeModal) closeNotice(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeNotice(); });

sendBtn.addEventListener('click', () => sendQuery(queryInput.value));

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery(queryInput.value);
  }
});

queryInput.addEventListener('input', () => {
  autoResize(queryInput);
  sendBtn.disabled = !queryInput.value.trim() || state.isLoading;
});

newChatBtn.addEventListener('click', () => { newConversation(); closeSidebar(); });

// 移动端侧栏
function openSidebar() { sidebar.classList.add('open'); sidebarMask.classList.add('show'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarMask.classList.remove('show'); }
document.getElementById('menuBtn').addEventListener('click', openSidebar);
sidebarMask.addEventListener('click', closeSidebar);
// .brand 元素已不存在（某次改版更名为 .intro-brand）；此前 null.addEventListener
// 在脚本加载时抛 TypeError，导致其后的 backHomeBtn 绑定从未执行——"返回首页"因此失灵。
// 可选链守卫：元素缺失时跳过，不再炸掉后续绑定。
document.querySelector('.brand')?.addEventListener('click', showHome);
document.getElementById('backHomeBtn').addEventListener('click', showHome);
