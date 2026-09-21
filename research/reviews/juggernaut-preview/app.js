'use strict';
const d = window.PREVIEW;
const select = document.querySelector('#model');
const nativeSelect = document.querySelector('#native');
const rows = document.querySelector('#rows');
const comparisons = [...d.models, ...d.juggernaut.filter(r => r.native_resolution === 1024)];
const nativeReferences = d.references.filter(r => r.native_resolution === 1024);
const realVis = nativeReferences.find(r => r.run.startsWith('realvis-'));
function option(select, model) {
  const item = document.createElement('option');
  item.value = model.run;
  item.textContent = model.title + (model.complete === false ? ' (pending)' : '');
  select.append(item);
}
comparisons.forEach(m => option(select, m));
nativeReferences.forEach(m => option(nativeSelect, m));
select.value = realVis?.complete ? 'juggernaut-hyper-tcd6-cfg1p5-native1024-v1-tiled' : 'sdxl-turbo-1step-v1';
if (realVis?.complete) nativeSelect.value = realVis.run;
document.querySelector('#gray').addEventListener('change', e => document.body.classList.toggle('gray', e.target.checked));
function element(tag, text) {
  const e = document.createElement(tag);
  if (text) e.textContent = text;
  return e;
}
function render() {
  rows.replaceChildren();
  const base = comparisons.find(m => m.run === select.value);
  const native = nativeReferences.find(r => r.run === nativeSelect.value);
  const lowResolution = d.juggernaut.find(r => r.native_resolution === 512);
  document.querySelector('#status').textContent = d.references.map(r => `${r.title}: ${r.entries.length}/8 images · ${r.complete ? 'complete' : 'result pending'}`).join(' · ');
  for (let i = 0; i < 8; i++) {
    const example = d.models[0].entries.find(e => e.index === i);
    const section = element('section');
    section.className = 'case';
    section.append(element('h2', `Case ${String(i).padStart(3, '0')} · seed ${example.seed}`));
    const grid = element('div');
    grid.className = 'grid';
    for (const model of [base, native, lowResolution]) {
      const entry = model.entries.find(e => e.index === i);
      const cell = element('div');
      cell.className = 'cell';
      cell.append(element('h3', model.title));
      if (entry) {
        const link = element('a');
        link.href = entry.files['native.png'].url;
        link.target = '_blank';
        link.rel = 'noopener';
        const image = element('img');
        image.src = entry.files['native.png'].url;
        image.alt = `${model.title}, case ${i}`;
        image.loading = 'lazy';
        link.append(image);
        cell.append(link);
      } else {
        const pending = element('div', 'Pending output');
        pending.className = 'missing';
        cell.append(pending);
      }
      grid.append(cell);
    }
    section.append(grid);
    const details = element('details');
    details.append(element('summary', 'Fixed prompt'), element('p', example.prompt));
    section.append(details);
    rows.append(section);
  }
}
select.addEventListener('change', render);
nativeSelect.addEventListener('change', render);
render();
