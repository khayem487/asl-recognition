import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createCameraSession,cameraError} from '../static/camera-session.mjs';
import {clock,deferred,fakeStream,flush} from './helpers/camera.mjs';
function harness(overrides={}) {
  const time=clock(),states=[],streams=[],calls={acquire:0,attach:[],detach:0,predict:[]};
  const options={
    acquire:async()=>{calls.acquire++;const s=fakeStream();streams.push(s);return s;},
    attach:async stream=>{calls.attach.push(stream);},detach:()=>{calls.detach++;},capture:()=> 'data:image/jpeg;base64,TEST',
    predict:async(frame,signal)=>{calls.predict.push({frame,signal});return {success:true,prediction:'A',mode:'model'};},
    onState:s=>states.push(s),now:time.now,setTimer:time.setTimer,clearTimer:time.clearTimer,...overrides};
  return {session:createCameraSession(options),time,states,streams,calls};
}

test('nothing acquires a camera or sends a frame until explicit Start; a pending permission request cannot duplicate',async()=>{
  const permission=deferred();let acquisitions=0;
  const h=harness({acquire:()=>{acquisitions++;return permission.promise;}});
  assert.equal(acquisitions,0);assert.equal(h.time.timers.size,0);assert.equal(h.session.getState().phase,'idle');
  const starting=h.session.start();assert.equal(h.session.getState().phase,'requesting');
  assert.equal(await h.session.start(),false);assert.equal(acquisitions,1);
  const stream=fakeStream();permission.resolve(stream);await starting;
  assert.equal(h.session.getState().active,true);assert.equal(h.calls.predict.length,0);
  h.session.stop();
});

test('Stop while permission is pending releases a late grant without attaching or uploading',async()=>{
  const permission=deferred(),h=harness({acquire:()=>permission.promise});
  const starting=h.session.start();h.session.stop('manual');
  const stream=fakeStream();permission.resolve(stream);assert.equal(await starting,false);
  assert.equal(stream.track.stops,1);assert.equal(h.calls.attach.length,0);assert.equal(h.calls.predict.length,0);
  assert.equal(h.session.getState().phase,'stopped');assert.equal(h.time.timers.size,0);
});

test('a pending old attachment cannot detach or overwrite a newer camera session',async()=>{
  const firstAttach=deferred();let attachments=0;
  const h=harness({attach:()=>++attachments===1?firstAttach.promise:Promise.resolve()});
  const first=h.session.start();await flush();const old=h.streams[0];
  h.session.stop();await h.session.start();const current=h.streams[1];
  firstAttach.resolve();assert.equal(await first,false);
  assert.equal(old.track.stops,1);assert.equal(current.track.stops,0);assert.equal(h.session.getState().active,true);
  h.session.stop();
});

test('inference requests are serial, and the next frame waits until the previous response plus interval',async()=>{
  const response=deferred();const requests=[];
  const h=harness({predict:(frame,signal)=>{requests.push({frame,signal});return response.promise;}});
  await h.session.start();await h.time.advance(0);assert.equal(requests.length,1);
  assert.equal(h.session.getState().phase,'warming');
  await h.time.advance(5000);assert.equal(requests.length,1);
  response.resolve({success:false,prediction:'NO_HAND',mode:'none'});await flush();
  assert.equal(h.session.getState().phase,'live');assert.equal(h.session.getState().result.prediction,'NO_HAND');
  await h.time.advance(1199);assert.equal(requests.length,1);
  await h.time.advance(1);assert.equal(requests.length,2);
  assert.ok(h.states.some(s=>s.phase==='analysing'));h.session.stop();
});

test('an old result after Stop and restart cannot change the new state or cancel its deadline',async()=>{
  const old=deferred(),current=deferred();let calls=0;
  const h=harness({predict:()=>++calls===1?old.promise:current.promise});
  await h.session.start();await h.time.advance(0);h.session.stop();
  await h.session.start();await h.time.advance(0);
  old.resolve({success:true,prediction:'OLD',mode:'model'});await flush();
  assert.equal(h.session.getState().phase,'warming');assert.equal(h.session.getState().result,null);
  assert.equal(h.time.timers.size,1,'the current request deadline survives the old finally');
  current.resolve({success:true,prediction:'B',mode:'model'});await flush();
  assert.equal(h.session.getState().result.prediction,'B');h.session.stop();
});

test('a request deadline aborts the request and releases the camera without another upload',async()=>{
  let signal;
  const h=harness({predict:(_,value)=>{signal=value;return deferred().promise;},requestTimeoutMs:2000});
  await h.session.start();await h.time.advance(0);await h.time.advance(2000);
  assert.equal(signal.aborted,true);assert.equal(h.session.getState().error.code,'server_timeout');
  assert.equal(h.streams[0].track.stops,1);assert.equal(h.time.timers.size,0);assert.equal(h.session.getState().active,false);
  await h.time.advance(10000);assert.equal(h.time.timers.size,0);
});

test('transient failures retry at a bounded cadence and stop the camera after three failures',async()=>{
  let requests=0;const h=harness({predict:async()=>{requests++;throw new Error('TEST server failure');}});
  await h.session.start();await h.time.advance(0);assert.equal(h.session.getState().phase,'retrying');
  await h.time.advance(2499);assert.equal(requests,1);
  await h.time.advance(1);assert.equal(requests,2);assert.equal(h.session.getState().attempt,2);
  await h.time.advance(2500);assert.equal(requests,3);assert.equal(h.session.getState().error.code,'server');
  assert.equal(h.streams[0].track.stops,1);assert.equal(h.time.timers.size,0);
});

test('a synchronous prediction failure is handled and its deadline cannot later reject unattended',async()=>{
  const h=harness({predict:()=>{throw new Error('TEST sync failure');},maxFailures:1});
  await h.session.start();await h.time.advance(0);
  assert.equal(h.session.getState().error.code,'server');assert.equal(h.time.timers.size,0);
});

test('capture waits for a ready frame without uploading an empty image; canvas failure stops safely',async()=>{
  let ready=false;const h=harness({capture:()=>ready?'TEST':null});
  await h.session.start();await h.time.advance(0);assert.equal(h.calls.predict.length,0);
  await h.time.advance(299);assert.equal(h.calls.predict.length,0);
  ready=true;await h.time.advance(1);assert.equal(h.calls.predict.length,1);h.session.stop();
  const broken=harness({capture:()=>{throw new Error('TEST no context');}});
  await broken.session.start();await broken.time.advance(0);
  assert.equal(broken.session.getState().error.code,'capture');assert.equal(broken.streams[0].track.stops,1);
});

test('a disconnected camera stops the current session; a removed old track handler cannot stop a restart',async()=>{
  const h=harness();await h.session.start();const old=h.streams[0],oldHandler=[...old.listeners][0];
  old.track.ended();assert.equal(h.session.getState().reason,'camera_disconnected');
  assert.equal(old.track.stops,1);assert.equal(h.time.timers.size,0);
  await h.session.start();oldHandler();assert.equal(h.session.getState().active,true);
  assert.equal(h.streams[1].track.stops,0);h.session.stop();
});

test('playback rejection releases the acquired stream and keeps the controls restartable',async()=>{
  const h=harness({attach:async()=>{throw {name:'NotReadableError'};}});
  assert.equal(await h.session.start(),false);assert.equal(h.session.getState().error.code,'busy');
  assert.equal(h.streams[0].track.stops,1);assert.equal(h.session.getState().starting,false);
});

test('the startup deadline bounds an attachment that never settles and aborts its readiness listener',async()=>{
  let signal;const h=harness({attach:(_,value)=>{signal=value;return deferred().promise;},startupTimeoutMs:1200});
  const starting=h.session.start();await flush();assert.equal(h.session.getState().phase,'starting');
  await h.time.advance(1200);assert.equal(await starting,false);
  assert.equal(signal.aborted,true);assert.equal(h.streams[0].track.stops,1);
  assert.equal(h.session.getState().error.code,'video_timeout');assert.equal(h.time.timers.size,0);
});

test('camera errors distinguish permission, missing hardware, busy camera and unsupported contexts without exposing native details',()=>{
  for(const [name,code] of [['NotAllowedError','permission'],['NotFoundError','missing'],
    ['NotReadableError','busy'],['SecurityError','security'],['NotSupportedError','unsupported'],['TimeoutError','video_timeout']]) {
    const error=cameraError({name,message:'TEST private device detail'});assert.equal(error.code,code);
    assert.equal(error.message.includes('private device'),false);
  }
});
