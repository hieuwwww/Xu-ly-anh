async function postFiles(files) {
  const form = new FormData();
  for (const f of files) form.append('images', f);
  const res = await fetch('/upload', { method: 'POST', body: form });
  return res.json();
}

function makeThumb(title, imgSrc, kind='step'){
  const wrap = document.createElement('div');
  wrap.className = 'thumb-item';
  const img = document.createElement('img');
  img.className = 'thumb';
  img.src = imgSrc;
  img.title = title;
  img.onclick = ()=> window.open(imgSrc, '_blank');
  const caption = document.createElement('div');
  caption.className = 'meta small';
  caption.innerText = title;
  wrap.appendChild(img);
  wrap.appendChild(caption);
  return wrap;
}

document.getElementById('uploadBtn').addEventListener('click', async () => {
  const input = document.getElementById('files');
  if (!input.files.length) return alert('Chọn file trước');
  const info = await postFiles(input.files);
  if (info.error) return alert(info.error);
  window._session = info.session;
  window._files = info.files;
  document.getElementById('sessionInfo').innerText = `Session: ${info.session} — ${info.files.length} ảnh`;

  // show originals
  const originals = document.getElementById('originals');
  originals.innerHTML = '';
  for (const f of info.files){
    const url = `/uploads/${info.session}/${f}`;
    const thumb = makeThumb(f, url, 'original');
    // mark original image class for larger size
    thumb.querySelector('img').classList.add('original');
    originals.appendChild(thumb);
  }

  // run sequential steps automatically at max speed
  startAutoRun(info.session, info.files);
});

async function fetchImageBlob(url){
  const r = await fetch(url);
  if(!r.ok) throw new Error('fetch failed');
  return URL.createObjectURL(await r.blob());
}

let _autoState = {playing:false, timer:null, index:0, queue:[], delay:80};

function buildQueue(files){
  const q = [];
  const n = files.length;
  for(let i=0;i<n;i++) q.push({type:'kp', idx:i});
  for(let i=0;i<n-1;i++) q.push({type:'match', i:i, j:i+1});
  for(let i=0;i<n-1;i++) q.push({type:'warp', i:i, j:i+1});
  q.push({type:'final'});
  return q;
}

async function processItem(session, files, item){
  if(item.type === 'kp'){
    try{
      const blob = await fetchImageBlob(`/keypoints/${session}/${item.idx}`);
      const thumb = makeThumb(files[item.idx], blob, 'step');
      document.getElementById('kpThumbs').appendChild(thumb);
    }catch(e){
      const el=document.createElement('div'); el.className='card small'; el.innerText='KP failed'; document.getElementById('kpThumbs').appendChild(el);
    }
  }else if(item.type === 'match'){
    try{
      const blob = await fetchImageBlob(`/matches/${session}/${item.i}/${item.j}`);
      const thumb = makeThumb(`${files[item.i]} ↔ ${files[item.j]}`, blob, 'step');
      document.getElementById('matchThumbs').appendChild(thumb);
    }catch(e){
      const el=document.createElement('div'); el.className='card small'; el.innerText='Match failed'; document.getElementById('matchThumbs').appendChild(el);
    }
  }else if(item.type === 'warp'){
    try{
      const blob = await fetchImageBlob(`/warp/${session}/${item.i}/${item.j}`);
      const thumb = makeThumb(`${files[item.j]} → ${files[item.i]}`, blob, 'step');
      document.getElementById('warpThumbs').appendChild(thumb);
    }catch(e){
      const el=document.createElement('div'); el.className='card small'; el.innerText='Warp failed'; document.getElementById('warpThumbs').appendChild(el);
    }
  }else if(item.type === 'final'){
    try{
      const blob = await fetchImageBlob(`/stitch_fast/${session}`);
      const thumb = makeThumb('Stitched panorama', blob, 'step');
      document.getElementById('finalThumbs').appendChild(thumb);
    }catch(e){
      const el=document.createElement('div'); el.className='card small'; el.innerText='Stitch failed'; document.getElementById('finalThumbs').appendChild(el);
    }
  }
}

async function runner(session, files){
  if(!_autoState.queue.length) _autoState.queue = buildQueue(files);
  if(_autoState.index >= _autoState.queue.length){ _autoState.playing=false; return; }
  const current = _autoState.queue[_autoState.index];
  await processItem(session, files, current);
  _autoState.index++;
  if(_autoState.playing){
    _autoState.timer = setTimeout(()=> runner(session, files), _autoState.delay);
  }
}

function startAutoRun(session, files){
  document.getElementById('kpThumbs').innerHTML = '';
  document.getElementById('matchThumbs').innerHTML = '';
  document.getElementById('warpThumbs').innerHTML = '';
  document.getElementById('finalThumbs').innerHTML = '';
  _autoState = {playing:true, timer:null, index:0, queue: buildQueue(files), delay:80};
  runner(session, files);
}

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}
