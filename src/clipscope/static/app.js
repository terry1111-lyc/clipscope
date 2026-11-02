const $ = id => document.getElementById(id);
let busy = false;
let imageURLs = [];
const headers = () => $('token').value ? {Authorization: 'Bearer ' + $('token').value} : {};
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers(), ...options.headers}});
  if (!response.ok) {
    let detail; try { detail = (await response.json()).detail; } catch { detail = response.statusText; }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return response.json();
}
function element(tag, text, className) {
  const el = document.createElement(tag); if (text !== undefined) el.textContent = text;
  if (className) el.className = className; return el;
}
async function connect() {
  try {
    const data = await api('/api/status');
    $('model').textContent = data.model.architecture + ' / ' + data.model.pretrained;
    $('device').textContent = data.device.toUpperCase();
    $('count').textContent = data.index.count + ' images';
    $('connection').textContent = 'Model online · 本地推理'; $('light').classList.add('online');
  } catch(e) { $('connection').textContent = '连接失败 · 检查服务或令牌'; message(e.message, true); }
}
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function setBusy(value) { busy = value; document.querySelectorAll('#query-form button').forEach(b => b.disabled = value); $('latency').textContent = value ? 'ENCODING…' : 'READY'; }
async function renderGroup(data, title) {
  const group = element('section'); group.append(element('h3', title, 'group-title'));
  const grid = element('div', undefined, 'results-grid'); group.append(grid);
  await Promise.all(data.results.map(async item => {
    const card = element('article', undefined, 'card');
    const img = element('img'); img.alt = item.caption;
    const body = element('div', undefined, 'card-body'); body.append(element('p', item.caption, 'card-title'));
    const score = element('div', undefined, 'score-row'); score.append(element('span', '#' + String(item.rank).padStart(2,'0')), element('strong', item.score.toFixed(4)));
    const bar = element('div', undefined, 'score-bar'); const fill = element('i'); fill.style.width = Math.max(0, item.score) * 100 + '%'; bar.append(fill);
    body.append(score, bar); card.append(img, body); grid.append(card);
    const response = await fetch('/api/images/' + encodeURIComponent(item.id), {headers: headers()});
    if (response.ok) { const url = URL.createObjectURL(await response.blob()); imageURLs.push(url); img.src = url; }
    else img.alt = '图片加载失败：' + item.caption;
  }));
  return group;
}
async function search(compare = false) {
  if (busy) return;
  const text = $('query').value.trim(), file = $('upload').files[0], k = Number($('k').value);
  if (!text && !file) return message('请输入查询或选择参考图片。', true);
  if (compare && file) return message('提示词对比需要先移除参考图片。', true);
  if (file && file.size > 7.5 * 1024 * 1024) return message('请选择小于 7.5 MiB 的图片，为表单留出请求空间。', true);
  setBusy(true); message('模型正在编码查询并检索图库…');
  try {
    const send = ensemble => api('/api/search', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, k, ensemble, diversity:compare ? 0 : Number($('diversity').value)})});
    let datasets, titles;
    if (file) {
      const body = new FormData(); body.append('image', file); body.append('text', text); body.append('alpha', $('alpha').value); body.append('k', k);
      datasets = [await api('/api/search/image', {method:'POST', body})]; titles = [text ? '混合查询 · Image + Text' : '参考图片 · Image query'];
    } else if (compare) { datasets = await Promise.all([send(false), send(true)]); titles = ['原始查询 · Baseline', '提示词集成 · Ensemble']; }
    else { datasets = [await send($('ensemble').checked)]; titles = [$('ensemble').checked ? '提示词集成 · Ensemble' : '原始查询 · Baseline']; }
    imageURLs.forEach(URL.revokeObjectURL); imageURLs = []; $('results').replaceChildren(); $('results').classList.toggle('compare', compare);
    for (let i = 0; i < datasets.length; i++) $('results').append(await renderGroup(datasets[i], titles[i]));
    $('result-title').textContent = compare ? '同一查询，两种表示' : '检索结果';
    message(compare ? '两组均使用纯余弦排序。并发请求的耗时包含模型锁等待，不用于正式性能比较。' : datasets[0].results.length + ' 个结果 · 已按当前检索策略排序');
    setBusy(false); $('latency').textContent = datasets.map(x => x.elapsed_ms + ' ms').join(' / ');
  } catch(e) { setBusy(false); message(e.message, true); }
}
async function reports() {
  $('report-list').replaceChildren(element('p', '正在读取实验记录…'));
  try {
    const data = await api('/api/reports'); $('report-list').replaceChildren();
    if (!data.length) { $('report-list').append(element('p', '尚无实验结果。运行 clipscope suite 后刷新；此处不会生成模拟分数。')); return; }
    for (const run of data) {
      const card = element('article', undefined, 'report-card'); card.append(element('h3', run.name));
      card.append(element('p', `${run.direction} · ${run.device} · ${run.split} · n=${run.count} · ${run.ensemble ? 'ensemble' : 'baseline'}`, 'report-meta'));
      const table = element('table'), head = element('tr'); ['Metric', 'Mean', '95% CI'].forEach(t => head.append(element('th', t))); table.append(head);
      for (const [metric, v] of Object.entries(run.metrics)) { const row = element('tr'); [metric, v.mean.toFixed(4), v.ci95.map(x => x.toFixed(4)).join(' — ')].forEach(t => row.append(element('td', t))); table.append(row); }
      card.append(table, element('p', `p50 ${run.latency_ms.p50.toFixed(1)} ms / p95 ${run.latency_ms.p95.toFixed(1)} ms · ${run.latency_ms.scope}`, 'report-meta'));
      const detail = element('details'); detail.append(element('summary', '失败案例：查看最低 nDCG 查询'));
      const list = element('ol'); run.failures.forEach(q => list.append(element('li', `${q.text} → ${q.ranking[0].id} · 正确：${q.relevant.join(', ')}`))); detail.append(list); card.append(detail);
      const slices = element('details'); slices.append(element('summary', '切片分析：属性 / 组合 / 改写'));
      const sliceList = element('ul'); Object.entries(run.slices).forEach(([name, metrics]) => sliceList.append(element('li', `${name}: hit@1 ${metrics['hit@1'].mean.toFixed(3)} · nDCG@10 ${metrics['ndcg@10'].mean.toFixed(3)}`))); slices.append(sliceList); card.append(slices);
      $('report-list').append(card);
    }
  } catch(e) { $('report-list').replaceChildren(element('p', e.message)); }
}
document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tab,.panel').forEach(el => el.classList.remove('active')); button.classList.add('active'); $(button.dataset.tab).classList.add('active');
  if (button.dataset.tab === 'reports') reports();
}));
document.querySelectorAll('[data-query]').forEach(b => b.addEventListener('click', () => { $('query').value = b.dataset.query; }));
['alpha','diversity'].forEach(id => $(id).addEventListener('input', () => $(id+'-value').textContent = Number($(id).value).toFixed(2)));
$('query-form').addEventListener('submit', e => { e.preventDefault(); search(); });
$('compare').addEventListener('click', () => search(true));
$('upload').addEventListener('change', () => $('clear-image').hidden = !$('upload').files.length);
$('clear-image').addEventListener('click', () => { $('upload').value = ''; $('clear-image').hidden = true; });
$('connect').addEventListener('click', connect); $('refresh').addEventListener('click', reports); connect();
