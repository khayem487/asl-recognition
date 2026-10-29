import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {createCameraSession} from '../static/camera-session.mjs';
import {clock,deferred,fakeStream,flush} from './helpers/camera.mjs';

class Element {
  constructor(){this.listeners=new Map();this.dataset={};this.attrs={};this.textContent='';this.disabled=false;this.hidden=false;
    this.classes=new Set();this.classList={add:value=>this.classes.add(value),remove:value=>this.classes.delete(value)};}
  addEventListener(type,fn){if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(fn);}
  removeEventListener(type,fn){this.listeners.get(type)?.delete(fn);}
  emit(type){for(const fn of [...(this.listeners.get(type)||[])])fn({type});}
  setAttribute(name,value){this.attrs[name]=value;}
}
function harness({permission,play}={}) {
  const time=clock(),elements=new Map(),window=new Element(),document=new Element();
  const template=readFileSync(new URL('../templates/index.html',import.meta.url),'utf8');
  const ids=[...template.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);
  ids.forEach(id=>elements.set(id,new Element()));
  const video=elements.get('camera-video'),stream=fakeStream(),requests=[],mediaCalls=[],drawing=[];
  Object.assign(video,{videoWidth:1280,videoHeight:720,readyState:2,play:play||(()=>Promise.resolve()),pauses:0,pause(){this.pauses++;}});
  const canvas={getContext:()=>({drawImage:(...args)=>drawing.push(args)}),toDataURL:(type,quality)=>{
    assert.equal(type,'image/jpeg');assert.equal(quality,.75);return 'data:image/jpeg;base64,TEST';}};
  Object.assign(document,{hidden:false,getElementById:id=>elements.get(id),createElement:tag=>{assert.equal(tag,'canvas');return canvas;}});
  let session;
  const source=readFileSync(new URL('../static/script.js',import.meta.url),'utf8');
  assert.ok(source.startsWith("import {createCameraSession} from './camera-session.mjs';"));
  runInNewContext(source.replace(/^import[^\n]+\n/,''),{
    document,window,DOMException,AbortController,performance:{now:time.now},setTimeout:time.setTimer,clearTimeout:time.clearTimer,
    navigator:{mediaDevices:{getUserMedia:constraints=>{mediaCalls.push(constraints);return permission||Promise.resolve(stream);}}},
    createCameraSession:options=>(session=createCameraSession({...options,now:time.now,setTimer:time.setTimer,clearTimer:time.clearTimer})),
    fetch:(path,options)=>{const response=deferred();requests.push({path,options,response});return response.promise;}
  },{filename:'static/script.js'});
  return {time,elements,window,document,video,stream,requests,mediaCalls,canvas,drawing,session,
    element:id=>elements.get(id),
    async start(){elements.get('start-btn').emit('click');await flush();await time.advance(0);},
    async respond(index,data,{ok=true,type='application/json'}={}){
      requests[index].response.resolve({ok,headers:{get:()=>type},json:async()=>data});await flush();
    }};
}

test('actual UI is idle until Start and requests only video, with bounded 640px frame capture and visible first-response state',async()=>{
  const h=harness();assert.equal(h.mediaCalls.length,0);assert.equal(h.requests.length,0);
  assert.equal(h.element('stop-btn').disabled,true);assert.equal(h.element('start-btn').disabled,false);
  await h.start();assert.equal(h.mediaCalls.length,1);assert.equal(h.mediaCalls[0].audio,false);
  assert.equal(h.requests.length,1);assert.equal(h.requests[0].path,'/process_frame');
  assert.equal(h.canvas.width,640);assert.equal(h.canvas.height,360);
  assert.equal(h.element('status-title').textContent,'Waiting for the first prediction');
  assert.equal(h.element('session-status').dataset.busy,'true');assert.equal(h.element('stop-btn').disabled,false);
  h.session.stop();
});

test('model results are labelled as predictions without a fabricated confidence, then NO_HAND clears the old letter',async()=>{
  const h=harness();await h.start();await h.respond(0,{success:true,prediction:'B',mode:'model'});
  assert.equal(h.element('prediction').textContent,'B');assert.equal(h.element('mode').textContent,'Trained model');
  assert.equal(h.element('prediction-live').textContent,'Latest prediction: B');
  assert.equal(h.element('prediction').textContent.includes('%'),false);
  await h.time.advance(1200);await h.respond(1,{success:false,prediction:'NO_HAND',mode:'none'});
  assert.equal(h.element('prediction').textContent,'—');assert.match(h.element('result-note').textContent,/No hand landmarks/);
  h.session.stop();
});

test('fallback is displayed as hand presence and never as an ASL letter',async()=>{
  const h=harness();await h.start();await h.respond(0,{success:true,prediction:'HAND_DETECTED',mode:'fallback-opencv'});
  assert.equal(h.element('prediction').textContent,'Hand detected');assert.equal(h.element('mode').dataset.mode,'fallback');
  assert.match(h.element('result-note').textContent,/does not identify an ASL letter/);h.session.stop();
});

test('HTTP/non-JSON failures clear previous results and expose a retry status instead of silently continuing',async()=>{
  const h=harness();await h.start();await h.respond(0,{success:true,prediction:'C',mode:'model'});
  await h.time.advance(1200);await h.respond(1,null,{ok:false,type:'text/html'});
  assert.equal(h.element('prediction').textContent,'—');assert.equal(h.element('status-title').textContent,'Connection interrupted');
  assert.equal(h.element('prediction-live').textContent,'');assert.equal(h.requests.length,2);h.session.stop();
});

test('malformed or contradictory inference responses never become live predictions',async()=>{
  for(const data of [null,[],{success:true,prediction:'',mode:'model'},
    {success:true,prediction:'A'.repeat(65),mode:'model'},{success:true,prediction:'NO_HAND',mode:'model'},
    {success:false,prediction:'A',mode:'model'},{success:true,prediction:'A',mode:'unknown'},
    {success:true,prediction:'A',mode:'fallback'},{success:true,prediction:'A',mode:'model',error:'TEST'}]) {
    const h=harness();await h.start();await h.respond(0,data);
    assert.equal(h.element('status-title').textContent,'Connection interrupted');assert.equal(h.element('prediction').textContent,'—');h.session.stop();
  }
});

test('Stop aborts pending inference, releases tracks and ignores a late response through the real DOM wiring',async()=>{
  const h=harness();await h.start();h.element('stop-btn').emit('click');
  assert.equal(h.requests[0].options.signal.aborted,true);assert.equal(h.stream.track.stops,1);assert.equal(h.video.srcObject,null);
  assert.equal(h.element('start-btn').disabled,false);assert.equal(h.element('stop-btn').disabled,true);
  await h.respond(0,{success:true,prediction:'LATE',mode:'model'});
  assert.equal(h.element('prediction').textContent,'—');assert.equal(h.time.timers.size,0);
});

test('hiding or leaving the page stops capture, clears results, and never automatically restarts on return',async()=>{
  const h=harness();await h.start();await h.respond(0,{success:true,prediction:'A',mode:'model'});
  h.document.hidden=true;h.document.emit('visibilitychange');
  assert.equal(h.stream.track.stops,1);assert.equal(h.element('status-title').textContent,'Camera paused for privacy');
  assert.equal(h.element('prediction').textContent,'—');h.document.hidden=false;h.document.emit('visibilitychange');
  await h.time.advance(10000);assert.equal(h.mediaCalls.length,1);assert.equal(h.time.timers.size,0);
  const leaving=harness();await leaving.start();leaving.window.emit('pagehide');
  assert.equal(leaving.stream.track.stops,1);assert.equal(leaving.requests[0].options.signal.aborted,true);
});

test('permission denial is inline and restartable rather than an alert dialog',async()=>{
  const h=harness({permission:Promise.reject(new DOMException('TEST native detail','NotAllowedError'))});
  await h.start();assert.match(h.element('status-detail').textContent,/Camera access was not allowed/);
  assert.equal(h.element('start-btn').disabled,false);assert.equal(h.element('stop-btn').disabled,true);assert.equal(h.requests.length,0);
});

test('a never-settling video.play is bounded even when the video already has usable dimensions',async()=>{
  const playback=deferred(),h=harness({play:()=>playback.promise});
  h.element('start-btn').emit('click');await flush();
  assert.equal(h.video.readyState,2);assert.equal(h.element('status-title').textContent,'Starting your camera');
  await h.time.advance(12000);
  assert.equal(h.stream.track.stops,1);assert.equal(h.element('status-title').textContent,'Unable to continue');
  assert.match(h.element('status-detail').textContent,/camera did not become ready/i);
  assert.equal(h.element('start-btn').disabled,false);assert.equal(h.time.timers.size,0);
  playback.resolve();await flush();assert.equal(h.session.getState().active,false);
});
