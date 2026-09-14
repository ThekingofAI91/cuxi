// ===== 意见反馈 =====
(function bindFeedback() {
  const btn = document.getElementById('feedbackBtn');
  const modal = document.getElementById('feedbackModal');
  const closeBtn = document.getElementById('feedbackCloseBtn');
  if (!btn || !modal) return;
  function openFeedback() {
    const cid = (state && state.currentCharacter) || '';
    document.getElementById('fbChar').value = cid ? cid : '（未选择）';
    document.getElementById('fbContent').value = '';
    document.getElementById('fbContact').value = '';
    document.getElementById('fbStatus').textContent = '';
    modal.hidden = false;
  }
  function closeFeedback() { modal.hidden = true; }
  btn.addEventListener('click', openFeedback);
  if (closeBtn) closeBtn.addEventListener('click', closeFeedback);
  modal.addEventListener('click', e => { if (e.target === modal) closeFeedback(); });
  const submit = document.getElementById('fbSubmitBtn');
  submit.addEventListener('click', async () => {
    const content = document.getElementById('fbContent').value.trim();
    if (!content) { toast('说点啥再提交吧～'); return; }
    submit.disabled = true; submit.textContent = '提交中…';
    try {
      const resp = await fetch('/admin/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content,
          character: (state && state.currentCharacter) || '',
          contact: document.getElementById('fbContact').value.trim(),
          page: location.pathname,
        }),
      });
      if (!resp.ok) throw new Error('提交失败 ' + resp.status);
      toast('反馈已收到，感谢！');
      closeFeedback();
    } catch (e) {
      document.getElementById('fbStatus').textContent = '提交失败：' + e.message;
    } finally {
      submit.disabled = false; submit.textContent = '提交反馈';
    }
  });
})();
