// Real deployed UI and server history restoration, no mocked API responses.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=path.join(root,'evaluation','chat_experience_v4');
 const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:960}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:7860');await page.getByText('模型已连接').waitFor();
 await page.locator('#mode').selectOption('guarded');
 await page.locator('#text').fill('我的测试编号是青竹84。');
 const firstResponse=page.waitForResponse(r=>r.url().endsWith('/api/chat')&&r.status()===200);
 await page.locator('#send').click();const first=await (await firstResponse).json();await page.locator('.assistant .meta').waitFor({timeout:180000});
 await page.request.post('http://127.0.0.1:7860/api/reset',{data:{session_id:first.session_id}});
 await page.reload();await page.getByText('模型已连接').waitFor();await page.locator('.assistant .meta').waitFor();
 await page.locator('#text').fill('我刚才的测试编号是什么？');await page.locator('#send').click();await page.locator('.assistant .meta').nth(1).waitFor({timeout:180000});
 const recall=await page.locator('.assistant .bubble').nth(1).innerText();if(!recall.includes('青竹84'))throw Error('Restored source recall failed: '+recall);
 await page.locator('#newChat').click();await page.locator('#file').setInputFiles(path.join(root,'evaluation/images/face/public_astronaut.png'));
 await page.locator('#send').click();await page.locator('.thinking-dots').waitFor();await page.screenshot({path:path.join(out,'live-waiting.png')});
 await page.locator('.assistant .meta').waitFor({timeout:180000});const imageAnswer=await page.locator('.assistant .bubble').innerText();if(!imageAnswer.includes('宇航服'))throw Error(imageAnswer);
 if(await page.locator('.user .bubble img').count()!==1)throw Error('missing image');
 await page.screenshot({path:path.join(out,'live-desktop.png')});await page.reload();await page.getByText('模型已连接').waitFor();if(await page.locator('.user .bubble img').count()!==1)throw Error('image not persisted');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);await page.screenshot({path:path.join(out,'live-mobile.png')});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
 if(errors.length)throw Error(errors.join('\n'));
 const records={passed:true,realModel:true,expiredSessionRestored:true,recall,imageAnswer,errors};fs.writeFileSync(path.join(out,'live_ui.json'),JSON.stringify(records,null,2));console.log(JSON.stringify(records));
 // Release server sessions, retaining browser history until the test context closes.
 const ids=await page.evaluate(()=>chats.map(c=>c.session).filter(Boolean));for(const session_id of new Set(ids))await page.request.post('http://127.0.0.1:7860/api/reset',{data:{session_id}});
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
