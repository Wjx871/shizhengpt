// UI lifecycle regression with a clearly simulated model; live checks are separate.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
const http=require('http'),fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..'),out=path.join(root,'evaluation','chat_experience_v4');fs.mkdirSync(out,{recursive:true});
let restores=0;
const server=http.createServer((req,res)=>{
 if(req.method==='GET'&&['/','/app.js','/app.css'].includes(req.url)){res.setHeader('Content-Type',req.url.endsWith('.js')?'application/javascript':req.url.endsWith('.css')?'text/css':'text/html');return res.end(fs.readFileSync(path.join(root,'web',req.url==='/'?'index.html':req.url.slice(1))));}
 res.setHeader('Content-Type','application/json');
 if(req.url==='/api/health')return res.end(JSON.stringify({ready:true}));
 let raw='';req.on('data',b=>raw+=b);req.on('end',()=>{const data=JSON.parse(raw||'{}');
  if(req.url==='/api/restore'){restores++;return res.end(JSON.stringify({session_id:'restored-session'}));}
  if(req.url==='/api/reset')return res.end('{"ok":true}');
  if(req.url==='/api/chat')return setTimeout(()=>res.end(JSON.stringify({session_id:data.session_id||'simulated-session',answer:'## 测试回答\n\n'+('这是一段用于验证逐字显示、历史保存和安全渲染的模拟回答。'.repeat(12))+'\n\n<script>alert(1)</script>[危险链接](javascript:alert(1))',mode:data.mode,guard_status:'limited_answer',latency_sec:1.2})),900);
  res.statusCode=404;res.end('{}');
 });
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const url=`http://127.0.0.1:${server.address().port}`;
 const browser=await chromium.launch({headless:true});const context=await browser.newContext({viewport:{width:1440,height:960}});const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(url);await page.getByText('模型已连接').waitFor();
 await page.screenshot({path:path.join(out,'desktop-empty.png')});
 await page.locator('#text').fill('偶尔有点疲劳，睡眠质量一般');await page.locator('#send').click();
 await page.locator('.thinking-dots').waitFor();await page.screenshot({path:path.join(out,'desktop-waiting.png')});
 await page.locator('#finishTyping').waitFor({state:'visible'});const first=await page.locator('.assistant .markdown').innerText();await page.waitForTimeout(200);const next=await page.locator('.assistant .markdown').innerText();if(next.length<=first.length)throw Error('typewriter did not progress');
 await page.locator('#finishTyping').click();await page.locator('.assistant .meta').waitFor();
 if(await page.locator('.assistant script,.assistant a[href^="javascript:"]').count())throw Error('unsafe HTML');
 await page.screenshot({path:path.join(out,'desktop-conversation.png')});
 await page.reload();await page.getByText('模型已连接').waitFor();await page.locator('.assistant .meta').waitFor();
 if(await page.locator('.history-item').count()!==1)throw Error('history not persisted');
 await page.locator('#text').fill('继续');await page.locator('#send').click();await page.locator('#finishTyping').waitFor({state:'visible'});await page.locator('#finishTyping').click();await page.locator('.assistant .meta').nth(1).waitFor();if(restores!==1)throw Error('history not restored');
 await page.locator('#newChat').click();if(await page.locator('.row').count())throw Error('new chat did not clear view');
 await page.locator('.history-open').click();if(await page.locator('.row').count()!==4)throw Error('history could not reopen');
 await page.locator('#search').fill('not a match');if(await page.locator('.history-item').count())throw Error('search');await page.locator('#search').fill('');
 await page.locator('#mode').selectOption('guarded');if(await page.locator('.row').count())throw Error('mode did not start a new chat');
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(300);await page.locator('#toggleHistory').click();await page.locator('#sidebar.mobile-open').waitFor();await page.screenshot({path:path.join(out,'mobile-history.png')});await page.locator('#scrim').click({position:{x:350,y:300}});await page.waitForTimeout(300);await page.screenshot({path:path.join(out,'mobile.png')});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(overflow)throw Error('mobile overflow');
 await page.locator('#toggleHistory').click();page.once('dialog',d=>d.accept());await page.getByRole('button',{name:'删除聊天',exact:true}).click();await page.locator('.history-item').waitFor({state:'detached'});
 if(errors.length)throw Error(errors.join('\n'));
 console.log(JSON.stringify({passed:true,simulatedModel:true,persistence:true,resume:true,animation:true,typewriter:true,search:true,mobileOverflow:overflow,errors}));
 await browser.close();server.close();
})().catch(e=>{console.error(e);server.close();process.exit(1)});
