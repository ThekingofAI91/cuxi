// ============================================================
// 首页：名人全屏背景（一位名人对应一张背景图）
// ============================================================
const FALLBACK_LOCATIONS = {
  jung: { name: "波林根", lat: 47.2043, lon: 8.8994, note: "湖畔石塔 · 荣格自建隐居处" },
  adler: { name: "维也纳", lat: 48.2082, lon: 16.3738, note: "个体心理学发源地" },
  fengge: { name: "深圳三和", lat: 22.66, lon: 114.045, note: "三和大神 · 底层叙事现场" },
  zhangxuefeng: { name: "郑州大学", lat: 34.8176, lon: 113.5384, note: "给排水专业毕业 · 生涯起点" },
  wangyangming: { name: "龙场", lat: 26.847, lon: 106.59, note: "龙场悟道" },
};

// ============================================================
// 场景插画：放大特写后由地图渐变显现（手绘 SVG 扁平插画）
// ============================================================
const SCENE_ART = {
  jung: `<svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="scJgSky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0E1830"/><stop offset="1" stop-color="#2C4066"/></linearGradient>
<linearGradient id="scJgLake" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#33486C"/><stop offset="1" stop-color="#182844"/></linearGradient>
</defs>
<rect width="1600" height="640" fill="url(#scJgSky)"/>
<g fill="#E8E4D8"><circle cx="180" cy="120" r="3" opacity=".8"/><circle cx="420" cy="80" r="2.4" opacity=".6"/><circle cx="640" cy="150" r="2" opacity=".7"/><circle cx="980" cy="70" r="2.6" opacity=".65"/><circle cx="1420" cy="120" r="2.2" opacity=".7"/><circle cx="1180" cy="230" r="1.8" opacity=".5"/><circle cx="300" cy="260" r="1.8" opacity=".5"/><circle cx="760" cy="60" r="1.6" opacity=".55"/></g>
<circle cx="1280" cy="150" r="96" fill="#F1E4C2" opacity=".12"/><circle cx="1280" cy="150" r="52" fill="#F1E4C2"/>
<path d="M0 520 L260 400 L520 500 L820 380 L1120 490 L1400 410 L1600 480 L1600 640 L0 640 Z" fill="#1D2C4C"/>
<rect y="640" width="1600" height="260" fill="url(#scJgLake)"/>
<path d="M0 640 Q400 622 800 640 T1600 640 L1600 662 L0 662 Z" fill="#24344F"/>
<g>
<rect x="700" y="330" width="150" height="310" rx="10" fill="#8C7C64"/>
<g stroke="#6E6050" stroke-width="3" opacity=".5"><line x1="702" y1="392" x2="848" y2="392"/><line x1="702" y1="452" x2="848" y2="452"/><line x1="702" y1="512" x2="848" y2="512"/><line x1="702" y1="572" x2="848" y2="572"/></g>
<path d="M678 342 L775 226 L872 342 Z" fill="#4E443A"/>
<rect x="806" y="252" width="22" height="56" fill="#4E443A"/>
<circle cx="775" cy="402" r="48" fill="#F2C879" opacity=".14"/>
<rect x="756" y="376" width="38" height="52" rx="16" fill="#F2C879"/>
<rect x="756" y="470" width="38" height="52" rx="16" fill="#E8B968" opacity=".92"/>
<path d="M742 640 L742 572 Q742 546 775 546 Q808 546 808 572 L808 640 Z" fill="#4A3E32"/>
<rect x="866" y="452" width="96" height="188" fill="#7A6B56"/>
<path d="M850 462 L914 400 L978 462 Z" fill="#57493C"/>
<rect x="896" y="500" width="30" height="40" rx="12" fill="#E8B968" opacity=".85"/>
</g>
<g fill="#16281E"><path d="M380 640 L430 500 L480 640 Z"/><path d="M470 640 L512 532 L554 640 Z"/><path d="M1120 640 L1168 512 L1216 640 Z"/><path d="M1210 640 L1248 552 L1286 640 Z"/></g>
<g opacity=".2" fill="#F2C879"><rect x="760" y="660" width="30" height="120" rx="14"/><rect x="900" y="660" width="22" height="86" rx="10"/></g>
<g stroke="#4A6284" stroke-width="4" opacity=".5" stroke-linecap="round"><line x1="180" y1="700" x2="360" y2="700"/><line x1="520" y1="760" x2="700" y2="760"/><line x1="1000" y1="720" x2="1240" y2="720"/><line x1="360" y1="820" x2="600" y2="820"/><line x1="1150" y1="800" x2="1380" y2="800"/></g>
</svg>`,
  adler: `<svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="scAdSky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3A2C48"/><stop offset=".68" stop-color="#7A4E48"/><stop offset="1" stop-color="#96604C"/></linearGradient>
</defs>
<rect width="1600" height="900" fill="url(#scAdSky)"/>
<circle cx="800" cy="600" r="230" fill="#F0A05C" opacity=".2"/><circle cx="800" cy="600" r="130" fill="#F2B26E" opacity=".4"/>
<path d="M560 620 Q560 470 650 470 Q740 470 740 620 Z" fill="#3A2C3A"/>
<rect x="640" y="420" width="20" height="60" fill="#3A2C3A"/>
<path d="M820 620 L820 340 L880 180 L940 340 L940 620 Z" fill="#322634"/>
<path d="M852 186 L880 108 L908 186 Z" fill="#322634"/>
<circle cx="880" cy="100" r="9" fill="#E8C87E"/>
<rect x="852" y="380" width="56" height="70" rx="26" fill="#F0C27A" opacity=".8"/>
<g>
<rect x="50" y="420" width="210" height="200" fill="#4E3A3C"/><path d="M40 420 L155 350 L270 420 Z" fill="#3A2B30"/>
<rect x="290" y="470" width="180" height="150" fill="#5A4240"/><path d="M282 470 L380 412 L478 470 Z" fill="#42312F"/>
<rect x="980" y="450" width="200" height="170" fill="#523C3E"/><path d="M970 450 L1080 386 L1190 450 Z" fill="#3C2C30"/>
<rect x="1210" y="480" width="190" height="140" fill="#46343A"/><path d="M1202 480 L1305 424 L1408 480 Z" fill="#342630"/>
<rect x="1430" y="440" width="170" height="180" fill="#4E3A3C"/><path d="M1422 440 L1515 380 L1608 440 Z" fill="#3A2B30"/>
</g>
<g fill="#F0C27A"><rect x="80" y="460" width="26" height="34"/><rect x="130" y="460" width="26" height="34"/><rect x="180" y="460" width="26" height="34"/><rect x="105" y="525" width="26" height="34"/><rect x="205" y="525" width="26" height="34"/><rect x="320" y="505" width="24" height="30"/><rect x="368" y="505" width="24" height="30"/><rect x="416" y="505" width="24" height="30"/><rect x="1010" y="490" width="26" height="32"/><rect x="1060" y="490" width="26" height="32"/><rect x="1110" y="490" width="26" height="32"/><rect x="1035" y="550" width="26" height="32"/><rect x="1240" y="515" width="24" height="30"/><rect x="1288" y="515" width="24" height="30"/><rect x="1336" y="515" width="24" height="30"/><rect x="1460" y="480" width="24" height="32"/><rect x="1508" y="480" width="24" height="32"/><rect x="1556" y="480" width="24" height="32"/></g>
<rect y="620" width="1600" height="280" fill="#2A2028"/>
<path d="M700 900 L760 620 L840 620 L900 900 Z" fill="#352830"/>
<g><line x1="420" y1="620" x2="420" y2="468" stroke="#1E1620" stroke-width="10"/><circle cx="420" cy="456" r="36" fill="#F0C27A" opacity=".16"/><circle cx="420" cy="456" r="14" fill="#F2C87E"/></g>
<g><line x1="1180" y1="620" x2="1180" y2="468" stroke="#1E1620" stroke-width="10"/><circle cx="1180" cy="456" r="36" fill="#F0C27A" opacity=".16"/><circle cx="1180" cy="456" r="14" fill="#F2C87E"/></g>
<g fill="#191019"><circle cx="800" cy="668" r="26"/><path d="M762 900 L770 706 Q800 690 830 706 L838 900 Z"/></g>
</svg>`,
  fengge: `<svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="scFgSky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0B0F1C"/><stop offset="1" stop-color="#1A2234"/></linearGradient>
</defs>
<rect width="1600" height="900" fill="url(#scFgSky)"/>
<g>
<rect x="0" y="200" width="220" height="500" fill="#232A3C"/>
<rect x="240" y="120" width="200" height="580" fill="#2C3448"/>
<rect x="460" y="240" width="180" height="460" fill="#232A3C"/>
<rect x="660" y="90" width="230" height="610" fill="#2C3448"/>
<rect x="910" y="180" width="200" height="520" fill="#232A3C"/>
<rect x="1130" y="130" width="220" height="570" fill="#2C3448"/>
<rect x="1370" y="220" width="230" height="480" fill="#232A3C"/>
</g>
<g fill="#F0C27A" opacity=".85"><rect x="30" y="240" width="20" height="24"/><rect x="80" y="240" width="20" height="24"/><rect x="130" y="300" width="20" height="24"/><rect x="280" y="160" width="20" height="24"/><rect x="330" y="160" width="20" height="24"/><rect x="380" y="220" width="20" height="24"/><rect x="280" y="280" width="20" height="24"/><rect x="500" y="280" width="18" height="22"/><rect x="560" y="280" width="18" height="22"/><rect x="500" y="360" width="18" height="22"/></g>
<g fill="#9CC8E8" opacity=".7"><rect x="700" y="130" width="22" height="26"/><rect x="752" y="130" width="22" height="26"/><rect x="804" y="190" width="22" height="26"/><rect x="700" y="250" width="22" height="26"/><rect x="950" y="220" width="20" height="24"/><rect x="1000" y="220" width="20" height="24"/><rect x="1050" y="280" width="20" height="24"/><rect x="1170" y="170" width="22" height="26"/><rect x="1222" y="170" width="22" height="26"/><rect x="1274" y="230" width="22" height="26"/><rect x="1410" y="260" width="20" height="24"/><rect x="1460" y="260" width="20" height="24"/><rect x="1512" y="320" width="20" height="24"/></g>
<g>
<rect x="540" y="450" width="340" height="150" rx="14" fill="#E86A8A" opacity=".16"/>
<rect x="560" y="470" width="300" height="110" rx="10" fill="#E86A8A"/>
<text x="710" y="545" font-size="64" font-weight="700" fill="#FFF5F8" text-anchor="middle" font-family="sans-serif">三 和</text>
<rect x="944" y="360" width="182" height="370" rx="14" fill="#58C8C0" opacity=".14"/>
<rect x="960" y="380" width="150" height="330" rx="10" fill="#58C8C0"/>
<text x="1035" y="470" font-size="46" font-weight="700" fill="#F2FFFE" text-anchor="middle" font-family="sans-serif">日</text>
<text x="1035" y="540" font-size="46" font-weight="700" fill="#F2FFFE" text-anchor="middle" font-family="sans-serif">结</text>
<text x="1035" y="610" font-size="46" font-weight="700" fill="#F2FFFE" text-anchor="middle" font-family="sans-serif">大</text>
<text x="1035" y="680" font-size="46" font-weight="700" fill="#F2FFFE" text-anchor="middle" font-family="sans-serif">神</text>
<rect x="224" y="414" width="232" height="112" rx="14" fill="#F0B45A" opacity=".16"/>
<rect x="240" y="430" width="200" height="80" rx="10" fill="#F0B45A"/>
<text x="340" y="486" font-size="44" font-weight="700" fill="#FFF8EC" text-anchor="middle" font-family="sans-serif">网 吧</text>
</g>
<rect y="700" width="1600" height="200" fill="#12161F"/>
<line x1="0" y1="700" x2="1600" y2="700" stroke="#3A4458" stroke-width="4"/>
<g><line x1="480" y1="700" x2="480" y2="540" stroke="#0C1018" stroke-width="12"/><circle cx="480" cy="528" r="44" fill="#F0C27A" opacity=".14"/><circle cx="480" cy="528" r="16" fill="#F2C87E"/><path d="M420 700 L540 700 L640 900 L320 900 Z" fill="#F0C27A" opacity=".06"/></g>
<g fill="#0C1018"><rect x="1120" y="760" width="110" height="70" rx="6"/><circle cx="1260" cy="700" r="26"/><path d="M1226 900 L1232 730 Q1260 716 1288 730 L1300 820 L1256 830 L1252 900 Z"/></g>
</svg>`,
  zhangxuefeng: `<svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="scZxSky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6EB6DC"/><stop offset="1" stop-color="#C8E6EE"/></linearGradient>
</defs>
<rect width="1600" height="900" fill="url(#scZxSky)"/>
<circle cx="1340" cy="150" r="90" fill="#FFE9A8" opacity=".35"/><circle cx="1340" cy="150" r="54" fill="#FFE9A8"/>
<g fill="#FFFFFF" opacity=".9"><ellipse cx="320" cy="150" rx="110" ry="34"/><ellipse cx="420" cy="128" rx="80" ry="28"/><ellipse cx="900" cy="100" rx="120" ry="30"/><ellipse cx="1000" cy="122" rx="70" ry="24"/></g>
<rect y="640" width="1600" height="260" fill="#7CB86A"/>
<path d="M640 900 L720 640 L880 640 L960 900 Z" fill="#C8C0B4"/>
<g>
<path d="M540 300 L800 230 L1060 300 Z" fill="#8E3C2E"/>
<rect x="560" y="300" width="480" height="60" fill="#B8503E"/>
<rect x="600" y="360" width="44" height="280" fill="#B8503E"/>
<rect x="956" y="360" width="44" height="280" fill="#B8503E"/>
<rect x="640" y="380" width="320" height="84" fill="#F4EFE4"/>
<text x="800" y="440" font-size="54" font-weight="700" fill="#3A3A44" text-anchor="middle" letter-spacing="14" font-family="serif">郑州大学</text>
</g>
<g><rect x="330" y="560" width="26" height="80" fill="#7A5C40"/><circle cx="343" cy="520" r="66" fill="#5E9E50"/><rect x="1240" y="560" width="26" height="80" fill="#7A5C40"/><circle cx="1253" cy="520" r="66" fill="#5E9E50"/><rect x="180" y="590" width="22" height="50" fill="#7A5C40"/><circle cx="191" cy="560" r="44" fill="#6AAE5C"/><rect x="1400" y="590" width="22" height="50" fill="#7A5C40"/><circle cx="1411" cy="560" r="44" fill="#6AAE5C"/></g>
<g><path d="M1080 900 L1096 720 L1300 720 L1316 900 Z" fill="#5A4634"/><rect x="1076" y="700" width="244" height="26" rx="8" fill="#6E5640"/><line x1="1198" y1="700" x2="1180" y2="600" stroke="#2E2E36" stroke-width="8"/><circle cx="1176" cy="590" r="16" fill="#2E2E36"/></g>
</svg>`,
  wangyangming: `<svg viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="scWySky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#56707E"/><stop offset="1" stop-color="#B4C4CC"/></linearGradient>
</defs>
<rect width="1600" height="900" fill="url(#scWySky)"/>
<path d="M0 430 L300 260 L620 420 L940 250 L1260 410 L1600 280 L1600 900 L0 900 Z" fill="#8FA4B0"/>
<rect y="400" width="1600" height="60" fill="#E8EEF0" opacity=".2"/>
<path d="M0 560 L360 380 L700 540 L1080 370 L1420 530 L1600 460 L1600 900 L0 900 Z" fill="#6E8694"/>
<rect y="540" width="1600" height="54" fill="#E8EEF0" opacity=".22"/>
<path d="M0 700 L420 500 L820 680 L1240 490 L1600 660 L1600 900 L0 900 Z" fill="#52687A"/>
<path d="M0 900 L240 680 L560 830 L900 660 L1280 820 L1600 700 L1600 900 Z" fill="#3A4C5C"/>
<g fill="#2E4438"><path d="M180 900 L240 700 L300 900 Z"/><path d="M290 900 L338 760 L386 900 Z"/><path d="M1330 900 L1390 690 L1450 900 Z"/><path d="M1440 900 L1484 770 L1528 900 Z"/></g>
<g>
<path d="M600 900 Q600 690 800 690 Q1000 690 1000 900 Z" fill="#22303C"/>
<path d="M640 900 Q640 726 800 726 Q960 726 960 900 Z" fill="#141E28"/>
<circle cx="800" cy="820" r="120" fill="#F2C879" opacity=".1"/>
<g fill="#0E141C"><circle cx="800" cy="788" r="24"/><path d="M744 900 Q744 820 800 818 Q856 820 856 900 Z"/></g>
</g>
<g><rect x="1060" y="800" width="180" height="100" fill="#6E5A44"/><path d="M1030 806 L1150 720 L1270 806 Z" fill="#4A3C30"/><rect x="1128" y="836" width="44" height="64" fill="#3A2E24"/><rect x="1076" y="820" width="34" height="30" fill="#E8B968" opacity=".8"/></g>
<g stroke="#42566A" stroke-width="4" opacity=".6" stroke-linecap="round" fill="none"><path d="M380 240 Q400 222 420 240 Q440 222 460 240"/><path d="M500 200 Q518 184 536 200 Q554 184 572 200"/></g>
</svg>`,
};

// ============================================================
// 首页背景：人物详情使用全屏背景图
// 图片放 frontend/assets/bg/<id>.jpg（AI 生成），缺失时回退内置 SVG
// ============================================================
const BG_SRC = {
  home: 'assets/bg/home.jpg',
  jung: 'assets/bg/jung.jpg',
  adler: 'assets/bg/adler.jpg',
  fengge: 'assets/bg/fengge.jpg',
  zhangxuefeng: 'assets/bg/zhangxuefeng.jpg',
  wangyangming: 'assets/bg/wangyangming.jpg',
};
const _bgImgReady = {};
Object.keys(BG_SRC).forEach((id) => {
  const img = new Image();
  img.onload = () => { _bgImgReady[id] = true; };
  img.src = BG_SRC[id];
});

function detailBgShow(id) {
  if (!detailBg) return;
  if (_bgImgReady[id]) {
    detailBg.style.backgroundImage = "url('" + BG_SRC[id] + "')";
    detailBg.innerHTML = '';
    homeDetail.dataset.fallback = '0';
    return;
  }
  detailBg.style.backgroundImage = '';
  if (SCENE_ART && SCENE_ART[id]) {
    detailBg.innerHTML = SCENE_ART[id];
    homeDetail.dataset.fallback = '0';
  } else {
    detailBg.innerHTML = '';
    homeDetail.dataset.fallback = '1';
  }
}

// ============================================================
// 首页三段式流程：介绍 -> 选择人物 -> 人物详情 -> 进入对话
// ============================================================
const HOME_META = {
  jung: { ability: '解梦师', review: '把梦当作信使，陪你在潜意识里照见自己。' },
  adler: { ability: '恋爱顾问', review: '把爱看作合作，把自卑变成起点。' },
  wangyangming: { ability: '心学宗师', review: '破山中贼易，破心中贼难，专治想多做少。' },
  fengge: { ability: '街头观察者', review: '坏事里先找机会，是他的赢学。' },
  zhangxuefeng: { ability: '升学规划师', review: '普通家庭的孩子，选对路比努力更重要。' },
};

function homeMeta(c) {
  const m = HOME_META[c.id] || {};
  return { ability: c.ability || m.ability || '', review: c.review || m.review || '' };
}

function homeStage(name) {
  homeView.dataset.stage = name;
  // 四级垂直滑动：按 data-order 用 JS 驱动 transform，保证 intro→zone→select→detail 连贯过渡。
  // markup 里 data-order 是数字（0-3），这里按数值解析（原实现按名字查表得 undefined，
  // 导致所有 stage 都落入 else 分支——intro 被内联样式永久藏到屏幕外，页面无法点击）。
  const _order = { intro: 0, zone: 1, select: 2, detail: 3 };
  const active = _order[name];
  [homeIntro, homeZone, homeSelect, homeDetail].forEach((stage) => {
    if (!stage) return;
    const k = parseInt(stage.dataset.order, 10);
    const diff = active - k;
    if (diff === 0) {
      stage.style.transform = 'translateY(0)';
      stage.style.opacity = '1';
      stage.style.pointerEvents = 'auto';
      stage.setAttribute('aria-hidden', 'false');
    } else if (diff > 0) {
      stage.style.transform = 'translateY(-100%)';
      stage.style.opacity = '0';
      stage.style.pointerEvents = 'none';
      stage.setAttribute('aria-hidden', 'true');
    } else {
      stage.style.transform = 'translateY(100%)';
      stage.style.opacity = '0';
      stage.style.pointerEvents = 'none';
      stage.setAttribute('aria-hidden', 'true');
    }
  });
}

function homeCardHTML(c, i) {
  const meta = homeMeta(c);
  return `<div class="home-card" data-char-id="${escapeHtml(c.id)}" role="button" tabindex="0" aria-label="与 ${escapeHtml(c.name)} 对话">
      <div class="hc-ability">${escapeHtml(meta.ability)}</div>
      <div class="hc-name">${escapeHtml(c.name)}</div>
      <div class="hc-tagline">${escapeHtml(c.tagline || '')}</div>
      <div class="hc-review">${escapeHtml(meta.review)}</div>
    </div>`;
}

function renderDeck() {
  const all = state.characters.length > 0 ? state.characters : FALLBACK_CHARACTERS;
  const zone = state.homeZone;
  const list = zone ? all.filter(c => (c.zone || 'education') === zone) : all;
  state._homeList = list;
  if (homeCount) homeCount.textContent = list.length + ' 位';

  // 未选区域时（兜底，正常不会走到）按两区展示；选完区域后只渲染该区角色
  let html = '';
  if (!zone) {
    const edu = all.filter(c => (c.zone || 'education') === 'education');
    const ent = all.filter(c => (c.zone || 'education') === 'entertainment');
    if (edu.length) html += '<div class="deck-zone-label">问道</div>' + edu.map((c) => homeCardHTML(c)).join('');
    if (ent.length) html += '<div class="deck-zone-label">会心</div>' + ent.map((c) => homeCardHTML(c)).join('');
  } else {
    html = list.map((c) => homeCardHTML(c)).join('');
  }
  homeDeck.innerHTML = html;

  homeDeck.querySelectorAll('.home-card:not(.create-card-entry):not(.rt-entry-card)').forEach((el, i) => {
    el.style.transitionDelay = (i * 70) + 'ms';
    el.addEventListener('click', () => homeOpenDetail(el.dataset.charId));
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        homeOpenDetail(el.dataset.charId);
      }
    });
  });
  requestAnimationFrame(() => {
    homeDeck.querySelectorAll('.home-card').forEach((el) => el.classList.add('is-in'));
  });

  // 选完区域后：追加「圆桌会议」入口；娱乐区额外追加「创建人物」入口
  if (zone) {
    const rtEl = document.createElement('div');
    rtEl.className = 'home-card rt-entry-card';
    rtEl.dataset.rt = '1';
    rtEl.innerHTML = '<div class="hc-ability">多人同台</div><div class="hc-name">圆桌会议</div><div class="hc-tagline">让多位角色就同一话题交锋论道</div><div class="hc-review">跨时空的思想碰撞</div>';
    rtEl.addEventListener('click', () => openRoundtable());
    rtEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openRoundtable(); }
    });
    homeDeck.appendChild(rtEl);
    requestAnimationFrame(() => rtEl.classList.add('is-in'));

    if (zone === 'entertainment') {
      const createEl = document.createElement('div');
      createEl.className = 'home-card create-card-entry';
      createEl.dataset.charId = '__create__';
      createEl.innerHTML = '<div class="hc-name">＋ 创建人物</div>';
      createEl.addEventListener('click', openCreateModal);
      createEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openCreateModal(); }
      });
      homeDeck.appendChild(createEl);
      requestAnimationFrame(() => createEl.classList.add('is-in'));
    }
  }
}

function homeStart() {
  state.homeStarted = true;
  homeStage('zone');
}

function homeSelectZone(zone) {
  state.homeZone = zone;
  renderDeck();
  homeStage('select');
}

function homeOpenDetail(id) {
  const c = (state._homeList || []).find(x => x.id === id);
  if (!c) return;
  state.homeSelectedId = id;
  const meta = homeMeta(c);
  if (detailAbility) detailAbility.textContent = meta.ability;
  if (detailName) detailName.textContent = c.name;
  if (detailTagline) detailTagline.textContent = c.tagline || '';
  if (detailReview) detailReview.textContent = meta.review;
  if (detailDesc) detailDesc.textContent = c.description || '';
  const loc = c.location || FALLBACK_LOCATIONS[id];
  if (detailLoc) {
    detailLoc.innerHTML = loc ? ('<b>' + escapeHtml(loc.name || '') + '</b>' + escapeHtml(loc.note || '')) : '';
  }
  detailBgShow(id);
  // 详情页操作按钮：导出角色卡（所有角色）+ 删除（仅自建）。
  // 知识图谱改为后台自动构建，无需用户手动触发
  if (detailEnterBtn && detailEnterBtn.parentElement) {
    const hint = detailEnterBtn.parentElement;
    const oldDel = hint.querySelector('.detail-del');
    if (oldDel) oldDel.remove();
    const oldExp = hint.querySelector('.detail-export');
    if (oldExp) oldExp.remove();
    // 导出角色卡：下载原生 JSON 卡片（人设/世界书/示例对话/背景）
    const exp = document.createElement('button');
    exp.className = 'detail-del detail-export';
    exp.type = 'button';
    exp.textContent = '导出角色卡';
    exp.addEventListener('click', (e) => {
      e.stopPropagation();
      window.location.href = '/persona/characters/' + encodeURIComponent(c.id) + '/export';
    });
    hint.insertBefore(exp, detailEnterBtn);
    // 删除（仅自建）
    if (c.is_custom) {
      const del = document.createElement('button');
      del.className = 'detail-del';
      del.type = 'button';
      del.textContent = '删除角色';
      del.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteCustomCharacter(c.id, c.name);
      });
      hint.insertBefore(del, detailEnterBtn);
    }
  }
  homeStage('detail');
}

function homeBackToSelect() {
  homeStage('select');
  renderDeck();
}

function homeEnterChat() {
  if (!homeView || homeView.dataset.stage !== 'detail' || !state.homeSelectedId) return;
  if (state.isLoading) return;
  enterChat(state.homeSelectedId);
}

function renderHome() {
  if (!homeView || homeView.classList.contains('hidden')) return;
  if (homeView.dataset.stage === 'select') renderDeck();
}

if (introCtaBtn) introCtaBtn.addEventListener('click', homeStart);
if (homeIntro) {
  homeIntro.addEventListener('click', (e) => {
    if (e.target.closest('#homeNoticeBtn')) return;
    homeStart();
  });
}
document.querySelectorAll('#zoneDeck .zone-card').forEach((el) => {
  el.addEventListener('click', () => homeSelectZone(el.dataset.zone));
});
if (detailEnterBtn) {
  detailEnterBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    homeEnterChat();
  });
}
if (detailBackBtn) {
  detailBackBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    homeBackToSelect();
  });
}
if (zoneBackBtn) {
  zoneBackBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    homeStage('intro');
  });
}
if (selectBackHomeBtn) {
  selectBackHomeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    homeStage('zone');
  });
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && homeView && !homeView.classList.contains('hidden') && homeView.dataset.stage === 'detail') {
    homeBackToSelect();
  }
});

function showHome() {
  state.view = 'home';
  homeView.classList.remove('hidden');
  if (!state.homeStarted) {
    homeStage('intro');
  } else {
    homeStage('zone');
  }
  closeSidebar();
}

function enterChat(charId) {
  state.view = 'chat';
  homeView.classList.add('hidden');
  if (state.currentCharacter !== charId) {
    switchCharacter(charId);
  } else {
    applyTheme(themeFor(charId));
    updateEmptyState();
    renderConversations();
    renderMessages();
  }
  // 刷新恢复推迟到进入对话归属角色时才执行，避免首页后台发请求
  if (!state._pendingRecovered) {
    const conv = state.currentConversation;
    if (conv && (conv.messages || []).length > 0 && getConversationCharacter(conv) === charId) {
      state._pendingRecovered = true;
      recoverPendingAnswer();
    }
  }
}
