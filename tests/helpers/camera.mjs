import assert from 'node:assert/strict';
export function deferred() {
  let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});
  return {promise,resolve,reject};
}
export async function flush() {for(let i=0;i<12;i++)await Promise.resolve();}
export function clock() {
  let now=0,next=0;const timers=new Map();
  return {now:()=>now,timers,
    setTimer(fn,delay){const id=++next;timers.set(id,{at:now+delay,fn});return id;},
    clearTimer:id=>timers.delete(id),
    async advance(ms){
      const end=now+ms;
      for(let turns=0;turns<100;turns++){
        const ready=[...timers.entries()].filter(([,v])=>v.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];
        if(!ready){now=end;await flush();return;}
        now=ready[1].at;timers.delete(ready[0]);ready[1].fn();await flush();
      }
      throw new Error('Unexpected timer loop');
    }};
}
export function fakeStream() {
  const listeners=new Set();
  const track={stops:0,stop(){this.stops++;},addEventListener(type,fn){assert.equal(type,'ended');listeners.add(fn);},
    removeEventListener(type,fn){listeners.delete(fn);},ended(){for(const fn of [...listeners])fn();}};
  return {track,getTracks:()=>[track],getVideoTracks:()=>[track],listeners};
}
