'use strict';
const data = window.PREVIEW_DATA;
document.querySelector('#inference-note').textContent = data.inference_steps === 4 ? 'Four-step fallback after the primary one-step development failure. This is a separate inference configuration; it does not erase that failure.' : 'Primary one-step inference comparison.';
const otherPreview = document.querySelector('#other-preview');
otherPreview.href = data.inference_steps === 4 ? '../sdxl-pilot-preview/' : '../sdxl-pilot-4step-preview/';
otherPreview.textContent = data.inference_steps === 4 ? 'One-step comparison' : 'Four-step fallback comparison';
const checkpoint = document.querySelector('#checkpoint');
const rows = document.querySelector('#rows');
const viewer = document.querySelector('#viewer');
let selected = data.checkpoints.length - 1, caseIndex = 0, armIndex = 0;
const element = (tag, text, className) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (className) node.className = className; return node; };
for (const [index, run] of data.checkpoints.entries()) { const option = element('option', `${run.updates} updates · ${run.completed_cells}/96 images`); option.value = index; checkpoint.append(option); }
checkpoint.value = selected;
function openImage(row, arm) { if (!data.checkpoints[selected].cases[row].arms[arm].native) return; caseIndex = row; armIndex = arm; renderImage(); if (!viewer.open) viewer.showModal(); }
function renderImage() {
  const run = data.checkpoints[selected], row = run.cases[caseIndex], cell = row.arms[armIndex];
  document.querySelector('#image-title').textContent = `${run.updates} updates · ${row.case_id} · ${cell.label}`;
  const full = document.querySelector('#full'); full.src = cell.native.url; full.alt = `${row.case_id}, ${cell.label}`;
  document.querySelector('#original').href = cell.native.url;
  document.querySelector('#image-details').textContent = `Native 512px PNG · seed ${row.seed} · ${document.body.classList.contains('gray') ? 'grayscale display filter' : 'original RGB'} · arrow keys change arm / case`;
}
function render() {
  rows.replaceChildren(); const run = data.checkpoints[selected];
  document.querySelector('#status').textContent = `${run.completed_cells}/96 development images available at ${run.updates} updates. ${run.complete ? 'All 24 cases and all four arms included.' : 'Evaluation is incomplete; unavailable cells are shown explicitly.'} No quality verdict is inferred.`;
  for (const [index, row] of run.cases.entries()) {
    const tr = element('tr'), heading = element('th'); heading.scope = 'row'; heading.append(element('span', row.case_id), element('span', `Seed ${row.seed}`, 'seed'));
    const detail = element('details'); detail.append(element('summary', 'Prompt'), element('p', row.prompt, 'prompt')); heading.append(detail); tr.append(heading);
    for (const [arm, cell] of row.arms.entries()) {
      const td = element('td'); td.append(element('div', cell.label, 'arm-label'));
      if (cell.native) {
        const button = element('button', undefined, 'portrait'); button.type = 'button'; button.setAttribute('aria-label', `Inspect ${row.case_id}, ${cell.label}`);
        const image = element('img'); image.src = cell.native.url; image.width = 512; image.height = 512; image.loading = index === 0 ? 'eager' : 'lazy'; image.alt = `${row.case_id}, ${cell.label}`; button.append(image); button.addEventListener('click', () => openImage(index, arm)); td.append(button);
        const caption = element('div', undefined, 'caption'), link = element('a', 'Record'); link.href = cell.record.url; link.target = '_blank'; link.rel = 'noopener'; caption.append(link, document.createTextNode(' · native PNG · hash verified')); td.append(caption);
      } else { td.append(element('div', cell.status || 'Pending evaluation', 'missing')); }
      tr.append(td);
    }
    rows.append(tr);
  }
}
function move(rowDelta, armDelta) {
  const nextRow = (caseIndex + rowDelta + 24) % 24, nextArm = (armIndex + armDelta + 4) % 4;
  openImage(nextRow, nextArm);
}
checkpoint.addEventListener('change', () => { selected = Number(checkpoint.value); render(); });
document.querySelector('#grayscale').addEventListener('change', event => { document.body.classList.toggle('gray', event.target.checked); if (viewer.open) renderImage(); });
document.querySelector('#close').addEventListener('click', () => viewer.close());
document.querySelector('#previous').addEventListener('click', () => move(0, -1)); document.querySelector('#next').addEventListener('click', () => move(0, 1));
document.addEventListener('keydown', event => { if (!viewer.open) return; const mapping = {ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}; if (mapping[event.key]) { event.preventDefault(); move(...mapping[event.key]); } });
render();
