document.addEventListener("DOMContentLoaded", () => {
  const cameraFeed = document.getElementById("camera-feed");
  const startBtn = document.getElementById("start-btn");
  const resultBox = document.getElementById("result");
  const predictionEl = document.getElementById("prediction");
  const modeEl = document.getElementById("mode");

  let stream = null;
  let active = false;
  let timer = null;

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      video.style.width = "100%";
      video.style.height = "100%";

      cameraFeed.innerHTML = "";
      cameraFeed.appendChild(video);

      active = true;
      startBtn.textContent = "Camera Running";
      startBtn.disabled = true;

      timer = setInterval(() => processFrame(video), 1200);
    } catch (error) {
      alert("Unable to access webcam: " + error.message);
    }
  }

  async function processFrame(video) {
    if (!active || !video.videoWidth || !video.videoHeight) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");

    try {
      const response = await fetch("/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: imageData })
      });

      const data = await response.json();
      if (data.success) {
        resultBox.classList.remove("hidden");
        predictionEl.textContent = data.prediction || "-";
        modeEl.textContent = "mode: " + (data.mode || "unknown");
      }
    } catch (error) {
      console.error("Frame processing failed", error);
    }
  }

  startBtn.addEventListener("click", () => {
    if (!active) startCamera();
  });

  window.addEventListener("beforeunload", () => {
    active = false;
    if (timer) clearInterval(timer);
    if (stream) stream.getTracks().forEach((t) => t.stop());
  });
});
