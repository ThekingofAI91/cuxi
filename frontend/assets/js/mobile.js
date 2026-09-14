// ============================================================
// 移动端体验增强
//  1) 软键盘适配：iOS Safari 键盘弹起只缩小 visualViewport，
//     不改 window.innerHeight，导致 100dvh / fixed 容器仍按布局视口排布，
//     输入区被键盘整个盖住。这里把"被吃掉的高度"写成 --kb、
//     "可视高度"写成 --vvh，由 CSS 把布局收缩到可视区。
//  2) 侧栏抽屉左滑关闭：跟手 + 阈值判定，纵向滚动优先不误触。
// 注意：本文件在所有业务脚本之后加载，依赖全局 closeSidebar()。
// ============================================================
(function () {
  'use strict';

  var root = document.documentElement;
  var isCoarse = window.matchMedia
    ? window.matchMedia('(hover: none) and (pointer: coarse)').matches
    : ('ontouchstart' in window);

  /* ---------------- 1. 软键盘适配 ---------------- */
  var vv = window.visualViewport;
  if (vv) {
    var lastKb = 0;
    var syncViewport = function () {
      // 键盘高度 = 布局视口 - 可视视口 - 视口顶部偏移
      var kb = Math.max(0, Math.round(window.innerHeight - vv.height - vv.offsetTop));
      root.style.setProperty('--kb', kb + 'px');
      root.style.setProperty('--vvh', Math.round(vv.height) + 'px');
      // 阈值 80px：过滤 iOS 地址栏收缩造成的抖动，只对真键盘响应
      var open = kb > 80;
      root.classList.toggle('kb-open', open);
      if (open && !lastKb) {
        // 键盘刚弹出：等布局收缩完成后把会话滚到底，避免最后一条被盖住
        setTimeout(scrollChatToBottom, 260);
      }
      lastKb = open ? kb : 0;
    };
    vv.addEventListener('resize', syncViewport);
    vv.addEventListener('scroll', syncViewport);
    syncViewport();
  }

  function scrollChatToBottom() {
    var c = document.getElementById('chatContainer');
    if (c) c.scrollTop = c.scrollHeight;
  }

  /* ---------------- 2. 侧栏左滑关闭 ---------------- */
  var sidebar = document.getElementById('sidebar');
  if (sidebar && isCoarse) {
    var startX = 0, startY = 0, dx = 0;
    var tracking = false, locked = false, width = 0;

    var isOpen = function () { return sidebar.classList.contains('open'); };

    var close = function () {
      if (typeof window.closeSidebar === 'function') window.closeSidebar();
      else sidebar.classList.remove('open');
    };

    sidebar.addEventListener('touchstart', function (e) {
      if (!isOpen() || e.touches.length !== 1) return;
      tracking = true; locked = false; dx = 0;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
      width = sidebar.offsetWidth || 300;
      sidebar.style.transition = 'none'; // 跟手期间关掉过渡
    }, { passive: true });

    sidebar.addEventListener('touchmove', function (e) {
      if (!tracking) return;
      var mx = e.touches[0].clientX - startX;
      var my = e.touches[0].clientY - startY;
      if (!locked) {
        if (Math.abs(mx) < 6 && Math.abs(my) < 6) return;
        // 纵向为主 → 判定为列表滚动，放弃手势，把过渡还回去
        if (Math.abs(my) > Math.abs(mx)) {
          tracking = false;
          sidebar.style.transition = '';
          return;
        }
        locked = true;
      }
      dx = Math.min(0, mx); // 只跟随左滑
      sidebar.style.transform = 'translateX(' + dx + 'px)';
    }, { passive: true });

    var endDrag = function () {
      if (!tracking) return;
      tracking = false;
      sidebar.style.transition = ''; // 恢复 CSS 过渡

      var threshold = Math.min(64, width / 3);
      if (dx < -threshold) {
        // 先动画滑出到位，再摘掉 .open 并清 inline transform（位置不变，无跳变）
        sidebar.style.transform = 'translateX(-100%)';
        setTimeout(function () {
          close();
          sidebar.style.transform = '';
        }, 260);
      } else {
        // 回弹
        sidebar.style.transform = '';
      }
      dx = 0;
    };

    sidebar.addEventListener('touchend', endDrag, { passive: true });
    sidebar.addEventListener('touchcancel', endDrag, { passive: true });
  }

  /* ---------------- 3. 杂项 ---------------- */
  // 老浏览器不支持 dvh 时用 JS 兜底（现代浏览器走不到这里）
  if (typeof CSS === 'undefined' || !CSS.supports || !CSS.supports('height', '100dvh')) {
    var setVh = function () {
      root.style.setProperty('--fallback-vh', window.innerHeight + 'px');
    };
    window.addEventListener('resize', setVh);
    setVh();
  }
})();
