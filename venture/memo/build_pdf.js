// Prints memo.html to Leadhaul_preseed_memo.pdf (US Letter) with headless Chromium.
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.join(__dirname, 'memo.html'));
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({ path: path.join(__dirname, 'Leadhaul_preseed_memo.pdf'), format: 'Letter', printBackground: true, preferCSSPageSize: true });
  await browser.close();
})();
