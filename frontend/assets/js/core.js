"use strict";
// ============================================================
// 状态
// ============================================================
const RT_MAX = 4; // 圆桌人数上限：人太多会削弱交锋感
const state = {
  sessionId: crypto.randomUUID(),
  conversations: [],
  currentConversation: null,
  isLoading: false,
  currentScene: 'persona',
  view: 'home',
  currentCharacter: 'jung',
  characters: [],
  _msgQueue: [],
  _animateFromIndex: 0,
  _pendingQuestions: [],
  _streamingText: '',
  _isStreaming: false,
  _gotToken: false,
  _reqStart: 0,
  _regenVariants: null,   // 重新生成时暂存的历史版本（重答完成后并入新消息）
  _ttsIdx: -1,            // 正在朗读的消息下标（-1 = 未在朗读）
};

const SUGGESTIONS = {
  jung: [
    "我反复梦见同一栋老房子，它想告诉我什么？",
    "什么是“阴影”？人为什么要面对它？",
    "人到中年感到迷茫空虚，您怎么看？",
    "怎样理解“个体化”这个过程？",
  ],
  adler: [
    "我总觉得自己不如别人，是怎么回事？",
    "童年经历对一个人的一生影响有多大？",
    "什么是“生活方式”？它是如何形成的？",
    "如何克服自卑、找回真正的勇气？",
  ],
};
const DEFAULT_SUGGESTIONS = [
  "您如何看待我此刻的困惑？",
  "请结合您的著作谈谈您的核心思想",
];

// DOM
const chatScroll = document.getElementById('chatScroll');
const chatContainer = document.getElementById('chatContainer');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const emptyState = document.getElementById('emptyState');
const newChatBtn = document.getElementById('newChatBtn');
const conversationList = document.getElementById('conversationList');
const charCards = document.getElementById('charCards');
const topbarPersona = document.getElementById('topbarPersona');
const topAvatar = document.getElementById('topAvatar');
const topName = document.getElementById('topName');
const topDesc = document.getElementById('topDesc');
const heroAvatar = document.getElementById('heroAvatar');
const heroMono = document.getElementById('heroMono');
const heroName = document.getElementById('heroName');
const heroDesc = document.getElementById('heroDesc');
const heroGreet = document.getElementById('heroGreet');
const suggestionChips = document.getElementById('suggestionChips');
const charHint = document.getElementById('charHint');
const sidebar = document.getElementById('sidebar');
const sidebarMask = document.getElementById('sidebarMask');
const brandMark = document.getElementById('brandMark');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const homeView = document.getElementById('homeView');
const homeZone = document.getElementById('homeZone');
const homeCount = document.getElementById('homeCount');
const homeIntro = document.getElementById('homeIntro');
const homeSelect = document.getElementById('homeSelect');
const homeDetail = document.getElementById('homeDetail');
const homeDeck = document.getElementById('homeDeck');
const introCtaBtn = document.getElementById('introCtaBtn');
const detailBg = document.getElementById('detailBg');
const detailBackBtn = document.getElementById('detailBackBtn');
const detailAbility = document.getElementById('detailAbility');
const detailName = document.getElementById('detailName');
const detailTagline = document.getElementById('detailTagline');
const detailReview = document.getElementById('detailReview');
const detailDesc = document.getElementById('detailDesc');
const detailLoc = document.getElementById('detailLoc');
const detailEnterBtn = document.getElementById('detailEnterBtn');
const selectBackHomeBtn = document.getElementById('selectBackHomeBtn');

// 圆桌会议（Roundtable）相关 DOM
const roundtableView = document.getElementById('roundTableView');
const rtBackBtn = document.getElementById('rtBackBtn');
const rtChips = document.getElementById('rtChips');
const rtMaxEl = document.getElementById('rtMax');
if (rtMaxEl) rtMaxEl.textContent = String(RT_MAX);
const rtTopic = document.getElementById('rtTopic');
const rtRounds = document.getElementById('rtRounds');
const rtStartBtn = document.getElementById('rtStartBtn');
const rtTranscript = document.getElementById('rtTranscript');
const rtEmpty = document.getElementById('rtEmpty');
const roundtableBtn = document.getElementById('roundtableBtn');
const introRoundtableBtn = document.getElementById('introRoundtableBtn');

// ============================================================
// 主题：按角色切换三种视觉风格（original / paper / noir）
// ============================================================
const THEME_MAP = {
  jung: 'original',
  adler: 'noir',
  fengge: 'street',
  zhangxuefeng: 'live',
  wangyangming: 'paper',
  plato: 'paper',
  aristotle: 'paper',
  socrates: 'noir',
  einstein: 'noir',
  marx: 'noir',
};
const THEMES = ['original', 'paper', 'noir', 'street', 'live'];

function themeFor(charId) {
  const c = state.characters.find(x => x.id === charId);
  if (c && THEMES.includes(c.theme)) return c.theme;
  return THEME_MAP[charId] || 'original';
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  if (brandMark) {
    brandMark.textContent = '';
  }
  const themeColorMeta = document.getElementById('themeColorMeta');
  if (themeColorMeta) {
    const bg = { original: '#161619', paper: '#EFE9DC', noir: '#181514', street: '#16181a', live: '#15171b' }[theme] || '#161619';
    themeColorMeta.content = bg;
  }
  const hint = document.getElementById('inputHint');
  if (hint) {
    hint.textContent = (theme === 'street' || theme === 'live') ? '对话基于公开言论生成' : '原著检索 · 引用可查';
  }
}

const FALLBACK_CHARACTERS = [
  { id: 'jung', name: '卡尔·荣格', avatar: '', description: '梦里乾坤，荣格解魂', tagline: '梦见老宅别乱猜，荣格开口梦全开。' },
  { id: 'adler', name: '阿尔弗雷德·阿德勒', avatar: '', description: '爱里迷路，阿德勒指路', tagline: '情场失意别发怵，阿德勒一聊心里有数。' },
];

// ============================================================
// 工具
// ============================================================
function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function formatTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

function formatConvTime(ts) {
  const d = new Date(ts);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  if (sameDay) return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  const yesterday = new Date(now.getTime() - 86400000);
  if (d.toDateString() === yesterday.toDateString()) return '昨天';
  return `${d.getMonth() + 1}月${d.getDate()}日`;
}

function scrollToBottom() {
  requestAnimationFrame(() => { chatContainer.scrollTop = chatContainer.scrollHeight; });
}

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 190) + 'px';
}

function currentChar() {
  return state.characters.find(c => c.id === state.currentCharacter) || null;
}

function getConversationCharacter(conv) {
  if (!conv) return null;
  // 优先按消息记录判断实际对话对象：消息中的角色头像是最可靠的证据
  // （characterId 可能被历史 bug 改写或复用逻辑重写，消息 sceneIcon 记录的是当时真实对话对象）
  for (const m of conv.messages || []) {
    const icon = m.extra?.sceneIcon;
    if (icon) {
      const found = state.characters.find(c => c.avatar === icon);
      if (found) return found.id;
    }
  }
  // 无消息或消息中无角色标记时，回退到 characterId
  const cid = conv.characterId;
  if (cid && state.characters.some(c => c.id === cid)) return cid;
  return null;
}

// 当前人物的对话列表（聊天界面不展示其他名人的会话）
function currentCharacterConvs() {
  return state.conversations.filter(c =>
    (!c.scene || c.scene === state.currentScene) &&
    (getConversationCharacter(c) || c.characterId) === state.currentCharacter
  );
}

// 后端权威核对：历史对话与谁对话，以后端记录为准（优先级最高）
// 后端只在会话"首次提问"时记录角色，不会被切换角色的操作改写，最可信。
// 点击历史对话 / 页面恢复时调用：若后端有记录且与本地判断不一致，
// 以后端为准修正并自愈本地数据（characterId + 消息 sceneIcon），
// 即使刷新/重启后本地判断也能稳定返回正确角色。
function verifyConversationOwner(conv) {
  if (!conv || !conv.sessionId) return;
  fetch('/conversation/' + encodeURIComponent(conv.sessionId) + '/meta')
    .then(r => r.json())
    .then(meta => {
      if (!meta || !meta.known || !meta.character) return; // 后端无记录：保持本地判断
      const local = getConversationCharacter(conv);
      if (meta.character === local) return; // 本地与后端一致：无需处理
      // 以历史对话的对象为最高优先级：后端权威覆盖本地（修复被污染的数据）
      conv.characterId = meta.character;
      const cdef = state.characters.find(c => c.id === meta.character) || null;
      if (cdef) {
        (conv.messages || []).forEach(m => {
          if (m.type === 'assistant') {
            m.extra = Object.assign({}, m.extra || {}, { sceneIcon: cdef.avatar });
          }
        });
      }
      if (state.currentConversation === conv) {
        state.currentCharacter = meta.character;
        applyTheme(themeFor(meta.character));
        updateEmptyState();
        charCards.querySelectorAll('.char-card').forEach(c =>
          c.classList.toggle('active', c.dataset.charId === meta.character)
        );
        renderMessages();
      }
      saveState();
      renderConversations();
    })
    .catch(() => { /* 后端核对失败不影响本地逻辑 */ });
}

// 姓名首字（印章式标记）：真人头像不可得时替代 emoji 头像
function nameMark(name) {
  const s = String(name || '');
  const parts = s.split('·');
  const last = parts[parts.length - 1].trim();
  return last ? last.charAt(0) : '?';
}

function getAssistantIcon() {
  return nameMark(currentChar()?.name);
}

// ============================================================
// 轻量 toast（多模块复用：反馈提示等）
// ============================================================
let _graphToastTimer = null;
function toast(msg, ms) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  if (_graphToastTimer) clearTimeout(_graphToastTimer);
  _graphToastTimer = setTimeout(() => el.classList.remove('show'), ms || 3000);
}
