// One explicit camera session, with at most one inference request at a time.
// Dependencies are injected so the lifecycle can be checked without a webcam.
export function cameraError(error) {
  const messages = {
    NotAllowedError: ['permission', 'Camera access was not allowed. Allow this site to use your camera, then try again.'],
    PermissionDeniedError: ['permission', 'Camera access was not allowed. Allow this site to use your camera, then try again.'],
    NotFoundError: ['missing', 'No camera was found. Connect a camera and try again.'],
    DevicesNotFoundError: ['missing', 'No camera was found. Connect a camera and try again.'],
    NotReadableError: ['busy', 'The camera could not be opened. Close another app using it, then try again.'],
    TrackStartError: ['busy', 'The camera could not be opened. Close another app using it, then try again.'],
    OverconstrainedError: ['settings', 'This camera does not support the requested settings. Try another camera.'],
    SecurityError: ['security', 'Camera access is blocked in this browser context. Open the HTTPS site directly.'],
    NotSupportedError: ['unsupported', 'This browser cannot open a camera here. Open the HTTPS site in a current browser.'],
    TimeoutError: ['video_timeout', 'The camera did not become ready. Check the connection, then try again.']
  };
  const [code, message] = messages[error?.name] || ['camera', 'The camera could not be started. Check its connection and try again.'];
  return {code, message};
}

export function createCameraSession({acquire, attach, detach, capture, predict, onState,
  now = () => performance.now(), setTimer = setTimeout, clearTimer = clearTimeout,
  intervalMs = 1200, retryMs = 2500, requestTimeoutMs = 45000, startupTimeoutMs = 12000, maxFailures = 3}) {
  let generation = 0, stream = null, starting = false, active = false;
  let nextTimer = null, timeoutTimer = null, startupTimer = null, requestController = null, startController = null;
  let firstResponse = true, failures = 0, trackListeners = [];
  let state = {phase: 'idle', active: false, starting: false, result: null};
  function emit(phase, extra = {}) {
    state = {phase, active, starting, result: null, ...extra};
    onState({...state});
  }
  function clearSchedule() {
    if (nextTimer !== null) clearTimer(nextTimer);
    if (timeoutTimer !== null) clearTimer(timeoutTimer);
    if (startupTimer !== null) clearTimer(startupTimer);
    nextTimer = timeoutTimer = startupTimer = null;
  }
  function release() {
    const previous = stream; stream = null;
    for (const [track, listener] of trackListeners) track.removeEventListener?.('ended', listener);
    trackListeners = [];
    previous?.getTracks().forEach(track => track.stop());
    detach();
  }
  function shutdown() {
    generation++; active = starting = false;
    clearSchedule();
    startController?.abort(); startController = null;
    requestController?.abort(); requestController = null;
    release();
  }
  function stop(reason = 'manual') {
    shutdown(); emit('stopped', {reason});
  }
  function schedule(token, delay = intervalMs) {
    if (!active || generation !== token) return;
    nextTimer = setTimer(() => {nextTimer = null; void process(token);}, delay);
  }
  async function process(token) {
    if (!active || generation !== token || requestController) return;
    let frame;
    try { frame = capture(); } catch {
      shutdown(); emit('error', {error: {code: 'capture', message: 'The camera frame could not be read. Start the camera again.'}}); return;
    }
    if (!frame) {
      emit('waiting_video'); schedule(token, 300); return;
    }
    const controller = new AbortController(); requestController = controller;
    const startedAt = now();
    emit(firstResponse ? 'warming' : 'analysing');
    let timedOut = false;
    const timeout = new Promise((_, reject) => {
      timeoutTimer = setTimer(() => {
        timedOut = true; controller.abort();
        reject(Object.assign(new Error('Request timed out'), {name: 'TimeoutError'}));
      }, requestTimeoutMs);
    });
    try {
      const result = await Promise.race([Promise.resolve().then(() => predict(frame, controller.signal)), timeout]);
      if (!active || generation !== token) return;
      firstResponse = false; failures = 0;
      emit('live', {result, elapsedMs: Math.max(0, Math.round(now() - startedAt))});
    } catch (error) {
      if (!active || generation !== token) return;
      failures++;
      if (timedOut || failures >= maxFailures) {
        const message = timedOut
          ? 'The recognition server took too long to respond. The camera has stopped; try again in a moment.'
          : 'The recognition server is unavailable. The camera has stopped; try again in a moment.';
        shutdown(); emit('error', {error: {code: timedOut ? 'server_timeout' : 'server', message}}); return;
      }
      emit('retrying', {attempt: failures, message: 'The server could not read this frame. Retrying shortly…'});
    } finally {
      if (requestController === controller) {
        if (timeoutTimer !== null) clearTimer(timeoutTimer);
        timeoutTimer = null; requestController = null;
      }
    }
    schedule(token, failures ? retryMs : intervalMs);
  }
  async function start() {
    if (starting || active) return false;
    const token = ++generation;
    starting = true; firstResponse = true; failures = 0;
    const controller = new AbortController(); startController = controller;
    let deadline = null;
    emit('requesting');
    try {
      const acquired = await acquire();
      if (generation !== token) {acquired.getTracks().forEach(track => track.stop()); return false;}
      stream = acquired;
      for (const track of stream.getVideoTracks()) {
        const ended = () => {if (generation === token) stop('camera_disconnected');};
        track.addEventListener?.('ended', ended); trackListeners.push([track, ended]);
      }
      emit('starting');
      // Bound the whole playback/readiness operation. A loaded video can still
      // have a pending play() promise, so readiness alone is not a deadline.
      await Promise.race([
        Promise.resolve().then(() => attach(stream, controller.signal)),
        new Promise((_, reject) => {
          deadline = startupTimer = setTimer(() => reject(
            Object.assign(new Error('Camera startup timed out'), {name:'TimeoutError'})
          ), startupTimeoutMs);
        })
      ]);
      if (generation !== token) return false;
      starting = false; active = true; startController = null;
      emit('waiting_video'); schedule(token, 0); return true;
    } catch (error) {
      if (generation !== token) return false;
      shutdown(); emit('error', {error: cameraError(error)}); return false;
    } finally {
      if (startupTimer === deadline && deadline !== null) {clearTimer(deadline);startupTimer = null;}
    }
  }
  return {start, stop, getState: () => ({...state})};
}
