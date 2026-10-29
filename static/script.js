import {createCameraSession} from './camera-session.mjs';

const $ = id => document.getElementById(id);
const video = $('camera-video'), startBtn = $('start-btn'), stopBtn = $('stop-btn');
const status = $('session-status'), statusTitle = $('status-title'), statusDetail = $('status-detail');
const prediction = $('prediction'), resultNote = $('result-note'), mode = $('mode');
const canvas = document.createElement('canvas');
let lastPrediction = null;

function waitForVideo(signal) {
  return new Promise((resolve, reject) => {
    const finish = error => {
      video.removeEventListener('loadeddata', ready); video.removeEventListener('resize', ready);
      signal.removeEventListener('abort', aborted);
      error ? reject(error) : resolve();
    };
    const ready = () => {if (video.videoWidth && video.videoHeight && video.readyState >= 2) finish();};
    const aborted = () => finish(new DOMException('Camera stopped', 'AbortError'));
    video.addEventListener('loadeddata', ready); video.addEventListener('resize', ready);
    signal.addEventListener('abort', aborted, {once:true});
    if (signal.aborted) aborted(); else ready();
  });
}

async function predict(frame, signal) {
  const response = await fetch('/process_frame', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({frame}), signal,
    credentials: 'same-origin', cache: 'no-store'
  });
  if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('Server unavailable');
  const data = await response.json();
  if (!data || typeof data !== 'object' || Array.isArray(data) || data.error || typeof data.success !== 'boolean' ||
      typeof data.prediction !== 'string' || !data.prediction.trim() || data.prediction.length > 64 || typeof data.mode !== 'string')
    throw new Error('Unrecognised inference response');
  const model = data.success && data.mode === 'model' && !['NO_HAND','HAND_DETECTED'].includes(data.prediction);
  const noHand = !data.success && data.mode === 'none' && data.prediction === 'NO_HAND';
  const fallback = data.success && ['fallback','fallback-opencv'].includes(data.mode) && data.prediction === 'HAND_DETECTED';
  if (!model && !noHand && !fallback) throw new Error('Unrecognised inference response');
  return data;
}

function showResult(data) {
  const label = data.prediction;
  prediction.dataset.kind = label.length === 1 ? 'letter' : 'message';
  if (data.success && data.mode === 'model' && label !== 'HAND_DETECTED') {
    const special = {space: 'Space', del: 'Delete', nothing: 'No sign'};
    prediction.textContent = special[label.toLowerCase()] || label;
    resultNote.textContent = label.toLowerCase() === 'nothing'
      ? 'The classifier returned its “nothing” class. Reposition your hand and try again.'
      : 'Latest model prediction. Hold the sign steady to compare the next frame.';
    mode.textContent = 'Trained model'; mode.dataset.mode = 'model';
  } else if (label === 'NO_HAND') {
    prediction.textContent = '—'; resultNote.textContent = 'No hand landmarks detected. Bring one hand into the frame.';
    mode.textContent = 'Waiting for a hand'; mode.dataset.mode = 'waiting';
  } else if (label === 'HAND_DETECTED') {
    prediction.textContent = 'Hand detected';
    resultNote.textContent = 'Hand-presence detection only. This response does not identify an ASL letter.';
    mode.textContent = 'Fallback detection'; mode.dataset.mode = 'fallback';
  } else {
    prediction.textContent = '—'; resultNote.textContent = 'No supported prediction was returned for this frame.';
    mode.textContent = 'Result unavailable'; mode.dataset.mode = 'waiting';
  }
  $('result').classList.add('has-result');
  if (lastPrediction !== prediction.textContent) $('prediction-live').textContent = 'Latest prediction: ' + prediction.textContent;
  lastPrediction = prediction.textContent;
}

function render(state) {
  const busy = state.starting || state.active;
  startBtn.disabled = busy; stopBtn.disabled = !busy;
  startBtn.textContent = state.starting ? 'Opening camera…' : state.active ? 'Camera running' : 'Start Camera';
  $('camera-feed').dataset.active = String(state.active || state.phase === 'starting');
  video.hidden = !(state.active || state.phase === 'starting');
  status.dataset.phase = state.phase;
  status.dataset.busy = String(['requesting','starting','warming','analysing'].includes(state.phase));
  const text = {
    idle: ['Ready when you are', 'Your camera is off. Start it when you are ready.'],
    requesting: ['Waiting for camera permission', 'Choose Allow in the browser prompt. Only video is requested.'],
    starting: ['Starting your camera', 'Waiting for the first video frame…'],
    waiting_video: ['Preparing a frame', 'Keep your hand inside the guide.'],
    warming: ['Waiting for the first prediction', 'The server may need time to wake up. Your camera can be stopped at any time.'],
    analysing: ['Camera is live', 'One frame is analysed at a time, about every 1.2 seconds after a response.'],
    live: ['Camera is live', 'One frame is analysed at a time, about every 1.2 seconds after a response.'],
    retrying: ['Connection interrupted', state.message || 'Retrying shortly…']
  };
  let [title, detail] = text[state.phase] || ['', ''];
  if (state.phase === 'stopped') {
    const reasons = {
      background: ['Camera paused for privacy', 'The page was hidden. Start the camera again when you return.'],
      camera_disconnected: ['Camera disconnected', 'Check the camera connection, then start again.'],
      navigation: ['Camera stopped', 'Video capture and pending analysis have stopped.']
    };
    [title, detail] = reasons[state.reason] || ['Camera stopped', 'Your camera is off. Start it again whenever you are ready.'];
  }
  if (state.phase === 'error') [title, detail] = ['Unable to continue', state.error.message];
  // Avoid repeatedly announcing the same live status between every pair of frames.
  if (statusTitle.textContent !== title) statusTitle.textContent = title;
  if (statusDetail.textContent !== detail) statusDetail.textContent = detail;
  if (state.phase === 'live' && state.result) {
    showResult(state.result);
    $('latency').textContent = 'Last request: ' + (state.elapsedMs / 1000).toFixed(1) + ' s';
  } else if (['idle','requesting','starting','stopped','error','retrying'].includes(state.phase)) {
    prediction.textContent = '—'; mode.textContent = 'No live prediction'; mode.dataset.mode = 'waiting';
    resultNote.textContent = state.phase === 'retrying' ? 'Waiting for a new successful response.' : 'Start the camera to see a prediction here.';
    $('latency').textContent = ''; $('prediction-live').textContent = ''; lastPrediction = null;
    $('result').classList.remove('has-result');
  }
}

const session = createCameraSession({
  acquire() {
    if (!navigator.mediaDevices?.getUserMedia) throw new DOMException('No media capture API', 'NotSupportedError');
    return navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:'user',width:{ideal:640},height:{ideal:480}}});
  },
  async attach(stream, signal) {
    video.srcObject = stream;
    // The session controller bounds this entire operation, including a pending play().
    await Promise.all([Promise.resolve().then(() => video.play()), waitForVideo(signal)]);
  },
  detach() {video.pause(); video.srcObject = null;},
  capture() {
    if (!video.videoWidth || !video.videoHeight || video.readyState < 2) return null;
    const scale = Math.min(1,640/video.videoWidth,480/video.videoHeight);
    canvas.width = Math.max(1,Math.round(video.videoWidth*scale)); canvas.height = Math.max(1,Math.round(video.videoHeight*scale));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Canvas unavailable');
    context.drawImage(video,0,0,canvas.width,canvas.height);
    return canvas.toDataURL('image/jpeg',0.75);
  }, predict, onState:render
});

startBtn.addEventListener('click', () => void session.start());
stopBtn.addEventListener('click', () => session.stop());
window.addEventListener('pagehide', () => session.stop('navigation'));
document.addEventListener('visibilitychange', () => {if (document.hidden) session.stop('background');});
render(session.getState());
