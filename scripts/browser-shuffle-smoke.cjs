const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('node:fs');
const http = require('node:http');
const assert = require('node:assert/strict');
const root = require('node:path').resolve(__dirname, '..');
(async () => {
  const app = fs.readFileSync(root + '/public/app.js');
  const sample = fs.readFileSync(root + '/public/portraits/01.webp');
  // Stand-in pixels isolate the real browser's storage/locking from generation progress.
  const server = http.createServer((req, res) => {
    if (req.url === '/app.js') { res.setHeader('Content-Type', 'text/javascript'); res.end(app); }
    else if (req.url.startsWith('/portraits/')) { res.setHeader('Content-Type', 'image/webp'); res.end(sample); }
    else { res.setHeader('Content-Type','text/html'); res.end('<img id="portrait"><script src="/app.js"></script>'); }
  }).listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({headless:true,channel:'chrome'});
  try {
    const context = await browser.newContext();
    let page = await context.newPage();
    const ids = [];
    for (let i = 0; i < 400; i++) {
      await page.goto(url);
      await page.waitForFunction(() => document.querySelector('#portrait').src.includes('.webp'));
      ids.push(await page.locator('#portrait').getAttribute('src'));
    }
    assert.equal(new Set(ids.slice(0,200)).size,200);
    assert.equal(new Set(ids.slice(200)).size,200);
    assert.notEqual(ids[199],ids[200]);
    const partial = [];
    for (let i = 0; i < 27; i++) {
      await page.goto(url);
      await page.waitForFunction(() => document.querySelector('#portrait').src.includes('.webp'));
      partial.push(await page.locator('#portrait').getAttribute('src'));
    }
    const state = await context.storageState({indexedDB:true});
    await context.close();
    const resumed = await browser.newContext({storageState:state});
    const tabs = await Promise.all(Array.from({length:20},()=>resumed.newPage()));
    const parallel = await Promise.all(tabs.map(async p=>{
      await p.goto(url);
      await p.waitForFunction(()=>document.querySelector('#portrait').src.includes('.webp'));
      return p.locator('#portrait').getAttribute('src');
    }));
    assert.equal(new Set(parallel).size,20);
    const remaining = await tabs[0].evaluate(async()=>{const db=await database;return new Promise(resolve=>{const req=db.transaction('queue').objectStore('queue').get('current');req.onsuccess=()=>resolve(req.result.remaining.length);});});
    assert.equal(new Set([...partial, ...parallel]).size,47);
    assert.equal(remaining,153);
    console.log(JSON.stringify({browser:'Chromium',reloads:427,uniquePerRound:200,concurrentTabs:20,distinctConcurrent:20,persistedState:true}));
  } finally { await browser.close(); server.close(); }
})().catch(e=>{console.error(e);process.exit(1)});
