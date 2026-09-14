// ============================================================
// 自建角色：创建 / 智能收集资料 / 删除
// ============================================================
const createModal = document.getElementById('createModal');
const cfName = document.getElementById('cfName');
const cfAvatar = document.getElementById('cfAvatar');
const cfTheme = document.getElementById('cfTheme');
const cfAbility = document.getElementById('cfAbility');
const cfTagline = document.getElementById('cfTagline');
const cfDesc = document.getElementById('cfDesc');
const cfBackground = document.getElementById('cfBackground');
const cfRolePrompt = document.getElementById('cfRolePrompt');
const cfFirstMes = document.getElementById('cfFirstMes');
const cfMesExample = document.getElementById('cfMesExample');
const cfWeb = document.getElementById('cfWeb');
const cfVerify = document.getElementById('cfVerify');
const cfResearchBtn = document.getElementById('cfResearchBtn');
const cfCreateBtn = document.getElementById('cfCreateBtn');
const cfCloseBtn = document.getElementById('createCloseBtn');
const cfStatus = document.getElementById('cfStatus');
const cfSources = document.getElementById('cfSources');
const createCharBtn = document.getElementById('createCharBtn');

function setCreateStatus(msg, type) {
  if (!cfStatus) return;
  cfStatus.textContent = msg || '';
  cfStatus.className = 'create-status' + (type ? ' ' + type : '');
}

function openCreateModal() {
  if (!createModal) return;
  createModal.hidden = false;
  createModal.classList.add('open');
  setCreateStatus('', '');
  cfSources.style.display = 'none';
  cfSources.innerHTML = '';
  if (cfName) cfName.focus();
}

function closeCreateModal() {
  if (!createModal) return;
  createModal.classList.remove('open');
  createModal.hidden = true;
}

async function researchCharacter() {
  const name = (cfName.value || '').trim();
  if (!name) { setCreateStatus('请先填写人物名字', 'err'); cfName.focus(); return; }
  cfResearchBtn.disabled = true;
  setCreateStatus('正在联网收集资料并生成人设草稿…', '');
  try {
    const resp = await fetch('/persona/characters/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        background: cfBackground.value || '',
        web_search: cfWeb.checked,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '收集失败');
    if (data.role_prompt) cfRolePrompt.value = data.role_prompt;
    if (data.background) cfBackground.value = data.background;
    // 酒馆式角色卡草稿自动回填（用户已手写的部分不覆盖）
    if (data.first_mes && !cfFirstMes.value.trim()) cfFirstMes.value = data.first_mes;
    if (data.mes_example && !cfMesExample.value.trim()) cfMesExample.value = data.mes_example;
    if (data.sources && data.sources.length > 0) {
      cfSources.innerHTML = '<div class="cs-title">已收集到的网络资料（将用于生成背景知识库）：</div><ul>'
        + data.sources.map(s => '<li>' + escapeHtml(s) + '</li>').join('') + '</ul>';
      cfSources.style.display = 'block';
      setCreateStatus('已生成人设草稿，可微调后点「创建」', 'ok');
    } else {
      cfSources.style.display = 'none';
      setCreateStatus('未搜到网络资料（可能无外网），已用你填写的背景生成；可手动补全后创建', 'err');
    }
  } catch (e) {
    setCreateStatus('收集失败：' + (e.message || e), 'err');
  } finally {
    cfResearchBtn.disabled = false;
  }
}

async function submitCreate() {
  const name = (cfName.value || '').trim();
  if (!name) { setCreateStatus('请填写人物名字', 'err'); cfName.focus(); return; }
  const background = (cfBackground.value || '').trim();
  const rolePrompt = (cfRolePrompt.value || '').trim();
  if (!background && !rolePrompt && !cfWeb.checked) {
    setCreateStatus('请至少填写「背景」或「角色人设」，或开启网络搜索', 'err');
    return;
  }
  cfCreateBtn.disabled = true;
  setCreateStatus('正在创建角色并构建知识库（首次检索需建索引，请稍候）…', '');
  try {
    const resp = await fetch('/persona/characters/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        background,
        role_prompt: rolePrompt || null,
        avatar: (cfAvatar.value || '').trim(),
        theme: cfTheme.value || 'original',
        ability: (cfAbility.value || '').trim(),
        tagline: (cfTagline.value || '').trim(),
        description: (cfDesc.value || '').trim(),
        use_web_search: cfWeb.checked,
        enable_verification: cfVerify.checked,
        first_mes: (cfFirstMes.value || '').trim(),
        mes_example: (cfMesExample.value || '').trim(),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || '创建失败');
    const newId = data.character.id;
    closeCreateModal();
    await loadCharacters();
    enterChat(newId);
  } catch (e) {
    setCreateStatus('创建失败：' + (e.message || e), 'err');
  } finally {
    cfCreateBtn.disabled = false;
  }
}

async function deleteCustomCharacter(id, name) {
  if (!id) return;
  if (!window.confirm(`确定删除自建角色「${name || id}」？\n该角色的知识库与对话记录将被清除，且不可恢复。`)) return;
  try {
    const resp = await fetch('/persona/characters/' + encodeURIComponent(id), { method: 'DELETE' });
    if (!resp.ok) {
      const d = await resp.json().catch(() => ({}));
      throw new Error(d.detail || '删除失败');
    }
    await loadCharacters();
    if (state.currentCharacter === id) {
      state.currentCharacter = 'jung';
      applyTheme(themeFor('jung'));
      updateEmptyState();
    }
  } catch (e) {
    alert('删除失败：' + (e.message || e));
  }
}

if (createCharBtn) createCharBtn.addEventListener('click', openCreateModal);
if (cfCloseBtn) cfCloseBtn.addEventListener('click', closeCreateModal);
if (createModal) {
  createModal.addEventListener('click', (e) => { if (e.target === createModal) closeCreateModal(); });
}
if (cfResearchBtn) cfResearchBtn.addEventListener('click', researchCharacter);
if (cfCreateBtn) cfCreateBtn.addEventListener('click', submitCreate);

// ---- 角色卡导入：选择 .json 卡片文件，原样交给后端导入并直接进入对话 ----
(function bindCardImport() {
  const btn = document.getElementById('cfImportBtn');
  const file = document.getElementById('cfImportFile');
  if (!btn || !file) return;
  btn.addEventListener('click', () => file.click());
  file.addEventListener('change', async () => {
    const f = file.files && file.files[0];
    file.value = '';
    if (!f) return;
    const isPng = /\.png$/i.test(f.name) || (f.type && f.type === 'image/png');
    let body;
    if (isPng) {
      // 社区通用 PNG 角色卡：原图 base64 交给后端解析文本块
      const buf = await f.arrayBuffer();
      let bin = '';
      const bytes = new Uint8Array(buf);
      for (let i = 0; i < bytes.length; i += 0x8000) {
        bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
      }
      body = { png_base64: btoa(bin) };
    } else {
      let card;
      try {
        card = JSON.parse(await f.text());
      } catch (e) {
        setCreateStatus('导入失败：这不是一个有效的 JSON 文件', 'err');
        return;
      }
      body = { card };
    }
    btn.disabled = true;
    setCreateStatus('正在导入角色卡…', '');
    try {
      const resp = await fetch('/persona/characters/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || '导入失败');
      const newId = data.character.id;
      closeCreateModal();
      await loadCharacters();
      enterChat(newId);
      toast(`已导入「${data.character.name}」，开始对话吧`);
    } catch (e) {
      setCreateStatus('导入失败：' + (e.message || e), 'err');
    } finally {
      btn.disabled = false;
    }
  });
})();
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && createModal && createModal.classList.contains('open')) closeCreateModal();
});
