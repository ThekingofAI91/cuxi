"use strict";
// ============================================================
// 首次运行设置页（开源版：让用户填自己的大模型 API）
// ============================================================
// 后端 GET /setup/status 返回 configured。未配置时把设置层拉起来盖住应用，
// 用户填完保存即生效（后端按配置文件 mtime 热加载，不需要重启服务）。
// 已配置时本文件不弹层，只注册侧栏的「模型设置」入口，方便随时改模型名。
//
// 注意：这里刻意不写死任何默认 Key 或私有中转地址，
// 默认值来自后端返回的 presets（公开服务商），保证开源后不泄露任何个人配置。

(function () {
  const overlay = document.getElementById('setupOverlay');
  if (!overlay) return;

  const baseUrlInput = document.getElementById('setupBaseUrl');
  const apiKeyInput = document.getElementById('setupApiKey');
  const modelInput = document.getElementById('setupModel');
  const presetsBox = document.getElementById('setupPresets');
  const statusEl = document.getElementById('setupStatus');
  const testBtn = document.getElementById('setupTestBtn');
  const saveBtn = document.getElementById('setupSaveBtn');
  const keyToggle = document.getElementById('setupKeyToggle');
  const setupBtn = document.getElementById('setupBtn');

  let presets = [];
  let busy = false;
  let configured = false;   // 本次加载时后端是否已有可用 Key

  // ---------- 基础 UI ----------
  function setStatus(text, kind) {
    statusEl.textContent = text || '';
    statusEl.className = 'setup-status' + (kind ? ' ' + kind : '');
  }

  function lock(state) {
    busy = state;
    testBtn.disabled = state;
    saveBtn.disabled = state;
  }

  function openOverlay() {
    overlay.hidden = false;
    document.documentElement.classList.add('setup-open');
    // 未填过 Key 时直接把光标放到 Key 输入框，少一次点击
    setTimeout(() => {
      if (!apiKeyInput.value) apiKeyInput.focus();
    }, 60);
  }

  function closeOverlay() {
    overlay.hidden = true;
    document.documentElement.classList.remove('setup-open');
  }

  function renderPresets() {
    presetsBox.textContent = '';
    presets.forEach((p) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'setup-preset';
      btn.textContent = p.label;
      btn.addEventListener('click', () => {
        baseUrlInput.value = p.base_url || '';
        if (p.model) modelInput.value = p.model;
        presetsBox.querySelectorAll('.setup-preset').forEach((el) => el.classList.remove('active'));
        btn.classList.add('active');
        setStatus('');
      });
      presetsBox.appendChild(btn);
    });
  }

  // 输入过程中清掉上一次的测试结论，避免"测过就是好的"误判
  [baseUrlInput, apiKeyInput, modelInput].forEach((el) => {
    el.addEventListener('input', () => {
      if (statusEl.classList.contains('ok') || statusEl.classList.contains('err')) setStatus('');
    });
  });

  keyToggle.addEventListener('click', () => {
    const show = apiKeyInput.type === 'password';
    apiKeyInput.type = show ? 'text' : 'password';
    keyToggle.textContent = show ? '隐藏' : '显示';
  });

  overlay.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !busy) {
      e.preventDefault();
      if (configured) save();
      else test();
    }
  });

  // ---------- 与后端交互 ----------
  async function loadStatus() {
    let data;
    try {
      const res = await fetch('/setup/status', { cache: 'no-store' });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      data = await res.json();
    } catch (e) {
      // 后端没起来也会走到这里：照样把设置层打开，让用户看到人话解释
      openOverlay();
      setStatus('连不上后端服务，请确认服务已启动后刷新页面', 'err');
      return;
    }

    presets = data.presets || [];
    renderPresets();
    configured = !!data.configured;

    if (data.base_url) baseUrlInput.value = data.base_url;
    if (data.model) modelInput.value = data.model;

    if (configured) {
      setStatus('当前已配置：' + (data.api_key_masked || '（来自 .env）'), 'ok');
    } else {
      openOverlay();
    }
  }

  function payload() {
    return {
      base_url: baseUrlInput.value.trim(),
      api_key: apiKeyInput.value.trim(),
      model: modelInput.value.trim(),
    };
  }

  async function post(url) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload()),
    });
    let body = {};
    try { body = await res.json(); } catch (e) { /* 空响应体 */ }
    return { res, body };
  }

  async function test() {
    if (busy) return;
    if (!payload().api_key && !configured) {
      setStatus('请先填写 API Key', 'err');
      return;
    }
    lock(true);
    setStatus('正在测试连接…');
    try {
      const { res, body } = await post('/setup/test');
      if (body.detail) {
        setStatus(body.detail, 'err');
      } else if (body.ok) {
        setStatus(body.message + '（' + body.latency_ms + 'ms）', 'ok');
      } else {
        setStatus(body.message || '连接失败', 'err');
      }
    } catch (e) {
      setStatus('测试请求发出失败：' + e.message, 'err');
    } finally {
      lock(false);
    }
  }

  async function save() {
    if (busy) return;
    if (!payload().api_key && !configured) {
      setStatus('请先填写 API Key', 'err');
      return;
    }
    lock(true);
    setStatus('正在保存…');
    try {
      const { res, body } = await post('/setup/llm');
      if (!res.ok) {
        setStatus(body.detail || '保存失败（HTTP ' + res.status + '）', 'err');
        return;
      }
      if (!configured) {
        // 首次配置：刷新一次让整个应用带着新配置重新起来，状态最干净
        setStatus('已保存，正在进入…', 'ok');
        location.reload();
        return;
      }
      configured = true;
      setStatus('已保存：' + (body.api_key_masked || '') + ' · ' + body.model, 'ok');
    } catch (e) {
      setStatus('保存请求发出失败：' + e.message, 'err');
    } finally {
      lock(false);
    }
  }

  testBtn.addEventListener('click', test);
  saveBtn.addEventListener('click', save);

  // 侧栏入口：随时回来改模型名 / 换 Key（后端只在已配置时校验权限）
  if (setupBtn) {
    setupBtn.addEventListener('click', () => {
      setStatus('');
      openOverlay();
      apiKeyInput.value = '';   // 不回显明文；留空表示"不修改"
      apiKeyInput.placeholder = configured ? '留空则沿用已保存的 Key' : 'sk-...';
      saveBtn.textContent = '保存';
      loadStatus();
    });
  }

  loadStatus();
})();
