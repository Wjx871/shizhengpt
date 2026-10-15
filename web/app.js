const $=id=>document.getElementById(id);
let current=null,chats=[],image=null,busy=false,ready=false,skipTyping=false,storage=null,saveQueue=Promise.resolve();
const restored=new Set();
function error(message){$('error').textContent=message;$('error').hidden=!message;}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,4000);}
function controls(){
 $('send').disabled=busy||!ready||(!$('text').value.trim()&&!image);
 ['text','attach','remove','newChat','sidebarNew','mode'].forEach(id=>$(id).disabled=busy);
 document.querySelectorAll('.history-open,.history-actions button').forEach(el=>el.disabled=busy);
}
function scroll(force=false){const el=$('messages');if(force||el.scrollHeight-el.scrollTop-el.clientHeight<160)el.scrollTop=el.scrollHeight;}
function clearImage(){image=null;$('file').value='';$('previewImg').removeAttribute('src');$('preview').hidden=true;controls();}
function updateModeHint(){$('mode').value=current?.mode||$('mode').value;document.querySelector('.hint').textContent=$('mode').value==='raw'?'原始模型 · 直接展示模型回答 · 仅供研究观察':'谨慎回答 · 保留可见观察与症状依据，医学结论仍需核实';}
function statusLabel(data){if(data.mode==='raw')return '原始模型';return ({image_observed:'图片观察',image_reviewed:'图片观察 · 已复核',image_uncertain:'图片细节待确认',image_abstained:'建议补拍',limited_answer:'依据核对',explanation_abstained:'需要补充信息',knowledge_abstained:'缺少用药依据',contradiction_abstained:'信息待核实',source_recall:'用户原文',informational:'问诊帮助'})[data.guard_status]||'谨慎回答';}
async function openStorage(){return new Promise((resolve,reject)=>{const request=indexedDB.open('shizhen-chats',1);request.onupgradeneeded=()=>request.result.createObjectStore('chats',{keyPath:'id'});request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});}
function dbOperation(kind,value){return new Promise((resolve,reject)=>{const tx=storage.transaction('chats',kind==='getAll'?'readonly':'readwrite');const store=tx.objectStore('chats');const req=kind==='getAll'?store.getAll():store[kind](value);let result;req.onsuccess=()=>result=req.result;tx.oncomplete=()=>resolve(result);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error);});}
async function persist(chat){if(!storage)return;const snapshot=JSON.parse(JSON.stringify(chat));saveQueue=saveQueue.catch(()=>{}).then(()=>dbOperation('put',snapshot));try{await saveQueue;}catch{toast('本地存储未成功，请释放浏览器空间；当前对话仍可继续。');}}
function makeChat(mode=$('mode').value){return {id:crypto.randomUUID(),mode,session:null,messages:[],title:'新对话',updated:Date.now()};}
function sidebar(){
 const q=$('search').value.trim().toLowerCase();$('history').replaceChildren();
 const items=chats.filter(c=>c.messages.length).sort((a,b)=>b.updated-a.updated);
 $('historyCount').textContent=items.length;
 for(const chat of items.filter(c=>(c.title+' '+c.messages.map(m=>m.text).join(' ')).toLowerCase().includes(q))){
  const item=document.createElement('div');item.className='history-item'+(chat.id===current?.id?' active':'');
  const open=document.createElement('button');open.className='history-open';open.disabled=busy;
  const title=document.createElement('div');title.className='history-title';title.textContent=chat.title;open.append(title);
  const date=document.createElement('div');date.className='history-date';date.textContent=new Date(chat.updated).toLocaleDateString('zh-CN',{month:'short',day:'numeric'})+' · '+(chat.mode==='raw'?'原始模型':'谨慎回答');open.append(date);
  open.onclick=()=>{if(!busy){current=chat;renderChat();closeMobile();}};
  const actions=document.createElement('div');actions.className='history-actions';
  for(const [label,symbol,handler] of [['重命名','✎',async()=>{const name=prompt('聊天名称',chat.title);if(name?.trim()){chat.title=name.trim().slice(0,80);await persist(chat);sidebar();}}],['删除聊天','×',async()=>{if(!confirm('删除这条聊天记录？'))return;chats=chats.filter(c=>c.id!==chat.id);if(storage){await saveQueue.catch(()=>{});try{await dbOperation('delete',chat.id);}catch{toast('删除失败，请稍后重试。');chats.push(chat);sidebar();return;}}if(chat.session)fetch('/api/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:chat.session})}).catch(()=>{});if(current?.id===chat.id){current=makeChat(chat.mode);renderChat();}else sidebar();}]] ){
   const button=document.createElement('button');button.type='button';button.title=label;button.setAttribute('aria-label',label);button.textContent=symbol;button.disabled=busy;button.onclick=()=>{if(!busy)handler();};actions.append(button);
  }
  item.append(open,actions);$('history').append(item);
 }
 if(!$('history').children.length){const empty=document.createElement('div');empty.className='history-empty';empty.textContent=q?'没有找到相关聊天':'聊过的内容，会留在这里。';$('history').append(empty);}
}
function row(role,text,img){$('welcome').hidden=true;const el=document.createElement('div');el.className='row '+role;if(role==='assistant'){const av=document.createElement('div');av.className='avatar';av.textContent='✳';el.append(av);}const bubble=document.createElement('div');bubble.className='bubble';if(img){const picture=document.createElement('img');picture.src=img;picture.alt='用户上传的图片';bubble.append(picture);}const body=document.createElement('div');body.textContent=text;bubble.append(body);el.append(bubble);$('messages').append(el);return {el,bubble,body};}
function metadata(view,message){const actions=document.createElement('div');actions.className='reply-actions';const meta=document.createElement('div');meta.className='meta';const d=message.reply||{};meta.textContent=statusLabel(d)+(d.latency_sec!=null?' · '+d.latency_sec+' 秒':'')+(d.truncated?' · 回答达到长度上限':'');const copy=document.createElement('button');copy.className='copy';copy.textContent='复制';copy.type='button';copy.onclick=async()=>{try{await navigator.clipboard.writeText(message.text);toast('已复制回答');}catch{toast('复制失败，请手动选择文字。');}};actions.append(meta,copy);view.bubble.append(actions);}
function renderChat(){document.querySelectorAll('.row').forEach(el=>el.remove());$('welcome').hidden=!!current.messages.length;$('mode').value=current.mode;$('text').value='';clearImage();error('');for(const message of current.messages){const view=row(message.role,message.text,message.image);if(message.role==='assistant'){renderAnswer(view.body,message.text);metadata(view,message);}}updateModeHint();sidebar();controls();scroll(true);}
function newChat(mode=$('mode').value){if(busy)return;current=makeChat(mode);restored.add(current.id);renderChat();closeMobile();$('text').focus();}
function closeMobile(){$('sidebar').classList.remove('mobile-open');$('scrim').hidden=true;}
function renderAnswer(element, text){
  const template=document.createElement('template');
  template.innerHTML=marked.parse(text,{gfm:true,breaks:true});
  const allowed=new Set('P BR STRONG EM DEL H1 H2 H3 H4 H5 H6 UL OL LI BLOCKQUOTE PRE CODE TABLE THEAD TBODY TR TH TD HR A'.split(' '));
  const blocked=new Set(['SCRIPT','STYLE','IFRAME','OBJECT','EMBED','SVG','MATH']);
  function clean(node){
    if(node.nodeType===Node.TEXT_NODE)return document.createTextNode(node.textContent);
    if(node.nodeType!==Node.ELEMENT_NODE||blocked.has(node.tagName))return document.createDocumentFragment();
    const output=allowed.has(node.tagName)?document.createElement(node.tagName.toLowerCase()):document.createDocumentFragment();
    if(node.tagName==='A'){
      try{const url=new URL(node.getAttribute('href'));if(['https:','http:'].includes(url.protocol)){output.href=url.href;output.target='_blank';output.rel='noopener noreferrer';}}catch{}
    }
    for(const child of node.childNodes)output.append(clean(child));
    return output;
  }
  const fragment=document.createDocumentFragment();
  for(const node of template.content.childNodes)fragment.append(clean(node));
  element.className='markdown';element.replaceChildren(fragment);
}
function waiting(view,hasImage){
 view.body.className='thinking';const dots=document.createElement('span');dots.className='thinking-dots';dots.innerHTML='<i></i><i></i><i></i>';
 const label=document.createElement('span');label.textContent=hasImage?'正在观察图片并生成回答':'正在生成回答';
 const time=document.createElement('span');time.className='thinking-time';view.body.replaceChildren(dots,label,time);
 const start=Date.now();const timer=setInterval(()=>{time.textContent=Math.floor((Date.now()-start)/1000)+' 秒';},500);return ()=>clearInterval(timer);
}
async function typeAnswer(element,text){
 skipTyping=matchMedia('(prefers-reduced-motion: reduce)').matches;$('finishTyping').hidden=skipTyping;
 const chars=Array.from(text),start=performance.now(),duration=Math.min(6500,Math.max(500,chars.length*9));
 while(!skipTyping){const progress=Math.min(1,(performance.now()-start)/duration);renderAnswer(element,chars.slice(0,Math.max(1,Math.ceil(chars.length*progress))).join(''));element.classList.add('typing-caret');scroll();if(progress>=1)break;await new Promise(r=>setTimeout(r,28));}
 renderAnswer(element,text);$('finishTyping').hidden=true;
}
async function request(path,payload){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let data;try{data=await response.json();}catch{throw Error('服务返回异常，请重试。');}if(!response.ok)throw Error(typeof data.error==='string'?data.error:data.error?.message||'请求失败');return data;}
async function ensureSession(chat){
 if(restored.has(chat.id)||!chat.messages.length)return;
 const data=await request('/api/restore',{session_id:chat.session,mode:chat.mode,messages:chat.messages.map(m=>({role:m.role,text:m.text,image:m.image||null}))});chat.session=data.session_id;restored.add(chat.id);
}
$('text').addEventListener('input',()=>{controls();$('text').style.height='auto';$('text').style.height=Math.min($('text').scrollHeight,160)+'px';});
$('text').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!$('send').disabled)$('form').requestSubmit();}});
$('attach').onclick=()=>$('file').click();$('remove').onclick=clearImage;
$('file').onchange=async()=>{
 const file=$('file').files[0];if(!file)return;
 if(!['image/jpeg','image/png','image/webp'].includes(file.type)||file.size>8*1024*1024){error('请选择8 MB以内的JPG、PNG或WebP图片。');$('file').value='';return;}
 $('attach').disabled=true;
 try{
  const source=await new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file);});
  // Keep the original upload bytes so history replay sees the same image.
  image=source;$('previewImg').src=image;$('fileName').textContent=file.name;$('preview').hidden=false;error('');
 }catch{error('图片读取失败。');}finally{controls();}
};
$('form').onsubmit=async e=>{
 e.preventDefault();if($('send').disabled)return;
 if(current.messages.length>=30){error('当前对话已达15轮，请新建对话。');return;}
 const chat=current,text=$('text').value.trim(),pic=image;busy=true;controls();error('');
 const user=row('user',text,pic),pending=row('assistant','');const stopWaiting=waiting(pending,!!pic);scroll(true);
 let committed=false;
 try{
  await ensureSession(chat);
  const payload={session_id:chat.session,mode:chat.mode,text,image:pic?pic.split(',')[1]:null};
  let data;
  try{data=await request('/api/chat',payload);}catch(ex){
   if(chat.messages.length&&/对话已过期/.test(ex.message)){restored.delete(chat.id);chat.session=null;await ensureSession(chat);payload.session_id=chat.session;data=await request('/api/chat',payload);}else throw ex;
  }
  stopWaiting();chat.session=data.session_id;restored.add(chat.id);
  const answer={role:'assistant',text:data.answer||'（模型返回了空回答）',reply:data};
  chat.messages.push({role:'user',text,image:pic||null},answer);chat.updated=Date.now();
  if(chat.messages.length===2&&chat.title==='新对话')chat.title=(text||'图片对话').slice(0,32);
  if(!chats.some(c=>c.id===chat.id))chats.push(chat);
  committed=true;await persist(chat);sidebar();$('text').value='';$('text').style.height='auto';clearImage();
  await typeAnswer(pending.body,answer.text);metadata(pending,answer);scroll();
 }catch(ex){
  if(!committed){user.el.remove();pending.el.remove();error(ex.message||'连接失败，请重试。');if(!chat.messages.length)$('welcome').hidden=false;}
  else{renderChat();toast('回答已保存，可在历史聊天中查看。');}
 }finally{stopWaiting();busy=false;$('finishTyping').hidden=true;controls();$('text').focus();}
};
$('finishTyping').onclick=()=>skipTyping=true;
$('newChat').onclick=()=>newChat();$('sidebarNew').onclick=()=>newChat();
$('mode').onchange=()=>newChat($('mode').value);
$('search').oninput=sidebar;
$('toggleHistory').onclick=()=>{if(innerWidth<=650){const open=$('sidebar').classList.toggle('mobile-open');$('scrim').hidden=!open;}else document.body.classList.toggle('history-hidden');};
$('scrim').onclick=closeMobile;
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeMobile();});
document.querySelectorAll('.chips button').forEach(button=>button.onclick=()=>{if(busy)return;$('text').value=button.textContent;controls();$('text').focus();});
(async()=>{
 current=makeChat();controls();updateModeHint();
 try{storage=await openStorage();chats=await dbOperation('getAll');chats=chats.filter(c=>c&&Array.isArray(c.messages)&&['raw','guarded'].includes(c.mode)).sort((a,b)=>b.updated-a.updated);if(chats.length)current=chats[0];}
 catch{toast('浏览器存储不可用，聊天记录仅在本次页面中保留。');}
 renderChat();
 try{const response=await fetch('/api/health');if(!response.ok)throw Error();const data=await response.json();ready=!!data.ready;$('statusText').textContent=ready?'模型已连接':'模型准备中';}
 catch{$('statusText').textContent='连接失败';error('无法连接模型服务，请检查服务器和SSH隧道，然后刷新页面。');}
 controls();
})();
