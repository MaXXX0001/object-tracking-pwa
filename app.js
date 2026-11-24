import * as ort from "./vendor/ort/ort.all.min.mjs";
import { ByteTracker } from "./tracker.js";

const MODEL_URL = "./models/yolo11n.onnx";
const DEMO_URL = "./demo/demo.mp4";
const PERSON_CHANNEL = 4;
const HIGH_THRESHOLD = 0.6;
const NMS_IOU = 0.7;
const MAX_CANDIDATES = 300;
const TRAIL_LENGTH = 40;
const MAX_VIEW_WIDTH = 960;
const FRAME_WAIT_MS = 50;

const $ = (id) => document.getElementById(id);
const video = $("source");
const view = $("view");
const viewContext = view.getContext("2d");
const scratch = document.createElement("canvas");
const scratchContext = scratch.getContext("2d", { willReadFrequently: true });

const state = {
  session: null,
  backend: null,
  running: false,
  generation: 0,
  facing: "environment",
  tracker: new ByteTracker(),
  uniqueIds: new Set(),
  trails: new Map(),
  frameIndex: 0,
  fps: 0,
  lastFrameTime: 0,
  lastVideoTime: -1,
  buffer: null,
};

function setStatus(message) {
  $("status").textContent = message;
}

function iou(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
  return union > 0 ? intersection / union : 0;
}

function nonMaxSuppression(candidates) {
  candidates.sort((a, b) => b[4] - a[4]);
  const kept = [];
  for (const candidate of candidates.slice(0, MAX_CANDIDATES)) {
    if (kept.every((other) => iou(candidate, other) < NMS_IOU)) kept.push(candidate);
  }
  return kept;
}

function inputBuffer(size) {
  if (state.buffer?.size !== size) {
    const input = new Float32Array(3 * size * size);
    const tensor = new ort.Tensor("float32", input, [1, 3, size, size]);
    state.buffer = { size, input, tensor };
  }
  return state.buffer;
}

function preprocess(width, height, size) {
  if (scratch.width !== size) {
    scratch.width = size;
    scratch.height = size;
  }
  const scale = size / Math.max(width, height);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const padX = Math.floor((size - drawWidth) / 2);
  const padY = Math.floor((size - drawHeight) / 2);
  scratchContext.fillStyle = "#727272";
  scratchContext.fillRect(0, 0, size, size);
  scratchContext.drawImage(video, padX, padY, drawWidth, drawHeight);
  const { data } = scratchContext.getImageData(0, 0, size, size);
  const { input, tensor } = inputBuffer(size);
  const area = size * size;
  for (let index = 0; index < area; index++) {
    input[index] = data[4 * index] / 255;
    input[area + index] = data[4 * index + 1] / 255;
    input[2 * area + index] = data[4 * index + 2] / 255;
  }
  return { tensor, scale, padX, padY };
}

function decode(output, scale, padX, padY, minScore) {
  const count = output.dims[2];
  const data = output.data;
  const candidates = [];
  for (let index = 0; index < count; index++) {
    const score = data[PERSON_CHANNEL * count + index];
    if (score < minScore) continue;
    const centerX = data[index];
    const centerY = data[count + index];
    const width = data[2 * count + index];
    const height = data[3 * count + index];
    candidates.push([
      (centerX - width / 2 - padX) / scale,
      (centerY - height / 2 - padY) / scale,
      (centerX + width / 2 - padX) / scale,
      (centerY + height / 2 - padY) / scale,
      score,
    ]);
  }
  return nonMaxSuppression(candidates);
}

async function createSession(backend) {
  ort.env.wasm.wasmPaths = new URL("./vendor/ort/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  return ort.InferenceSession.create(MODEL_URL, { executionProviders: [backend], graphOptimizationLevel: "all" });
}

async function ensureSession() {
  const wanted = $("backend").value;
  if (state.session && state.backend === wanted) return;
  setStatus(`Loading model (${wanted})...`);
  try {
    state.session = await createSession(wanted);
    state.backend = wanted;
  } catch (error) {
    if (wanted === "wasm") throw error;
    setStatus(`${wanted} failed (${error.message}); falling back to wasm`);
    $("backend").value = "wasm";
    state.session = await createSession("wasm");
    state.backend = "wasm";
  }
  setStatus(`Model ready on ${state.backend}${self.crossOriginIsolated ? "" : " (single thread: page is not cross-origin isolated)"}`);
}

function resetTracking() {
  state.tracker = new ByteTracker();
  state.uniqueIds = new Set();
  state.trails = new Map();
  state.frameIndex = 0;
  state.lastVideoTime = -1;
  $("unique").textContent = "-";
  $("people").textContent = "-";
}

function colorFor(id) {
  return `hsl(${(id * 47) % 360} 90% 55%)`;
}

function draw(tracked, width, height) {
  const viewWidth = Math.min(width, MAX_VIEW_WIDTH);
  const factor = viewWidth / width;
  const viewHeight = Math.round(height * factor);
  if (view.width !== viewWidth || view.height !== viewHeight) {
    view.width = viewWidth;
    view.height = viewHeight;
  }
  viewContext.drawImage(video, 0, 0, view.width, view.height);

  const showTrails = $("trails").checked;
  for (const { id, box } of tracked) {
    const [x1, y1, x2, y2] = box.map((value) => value * factor);
    const color = colorFor(id);
    const trail = state.trails.get(id) || { points: [], seen: 0 };
    trail.points.push([(x1 + x2) / 2, y2]);
    if (trail.points.length > TRAIL_LENGTH) trail.points.shift();
    trail.seen = state.frameIndex;
    state.trails.set(id, trail);

    if (showTrails && trail.points.length > 1) {
      viewContext.strokeStyle = color;
      viewContext.lineWidth = 2;
      viewContext.beginPath();
      trail.points.forEach(([x, y], index) => (index ? viewContext.lineTo(x, y) : viewContext.moveTo(x, y)));
      viewContext.stroke();
    }
    viewContext.strokeStyle = color;
    viewContext.lineWidth = 2;
    viewContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
    viewContext.font = "bold 14px system-ui";
    const label = `#${id}`;
    const labelWidth = viewContext.measureText(label).width + 8;
    viewContext.fillStyle = color;
    viewContext.fillRect(x1, y1 - 18, labelWidth, 18);
    viewContext.fillStyle = "#000";
    viewContext.fillText(label, x1 + 4, y1 - 4);
  }

  for (const [id, trail] of state.trails) {
    if (state.frameIndex - trail.seen > TRAIL_LENGTH) state.trails.delete(id);
  }
}

async function processFrame() {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;

  const size = Number($("size").value);
  const threshold = Number($("threshold").value);
  const started = performance.now();
  const { tensor, scale, padX, padY } = preprocess(width, height, size);
  const prepared = performance.now();
  const outputs = await state.session.run({ [state.session.inputNames[0]]: tensor });
  const inferred = performance.now();
  const output = outputs[state.session.outputNames[0]];
  const detections = decode(output, scale, padX, padY, threshold);
  output.dispose?.();

  const tracker = state.tracker;
  const tracked = tracker.update(detections);
  state.frameIndex += 1;
  tracked.forEach(({ id }) => state.uniqueIds.add(id));
  draw(tracked, width, height);
  const finished = performance.now();

  if (state.lastFrameTime) {
    const instant = 1000 / (finished - state.lastFrameTime);
    state.fps = state.fps ? state.fps * 0.8 + instant * 0.2 : instant;
  }
  state.lastFrameTime = finished;
  $("fps").textContent = state.fps.toFixed(1);
  $("latency").textContent = (inferred - prepared).toFixed(0);
  $("timing").textContent =
    `prep ${(prepared - started).toFixed(0)} ms · run ${(inferred - prepared).toFixed(0)} ms · post ${(finished - inferred).toFixed(0)} ms`;
  $("people").textContent = String(tracked.length);
  $("unique").textContent = String(state.uniqueIds.size);
}

// Waits for the next presented video frame so the browser can paint between iterations; the timeout keeps the loop alive if no frame arrives
function nextFrame() {
  return new Promise((resolve) => {
    let handle = 0;
    const timer = setTimeout(() => {
      video.cancelVideoFrameCallback?.(handle);
      resolve();
    }, FRAME_WAIT_MS);
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    if (video.requestVideoFrameCallback) handle = video.requestVideoFrameCallback(done);
    else requestAnimationFrame(done);
  });
}

async function loop(generation) {
  while (state.running && generation === state.generation) {
    await nextFrame();
    const isCurrent = state.running && generation === state.generation;
    const hasNewFrame = video.readyState >= 2 && !video.paused && video.currentTime !== state.lastVideoTime;
    if (!isCurrent || !hasNewFrame) continue;

    const looped = video.currentTime < state.lastVideoTime;
    if (looped) resetTracking();
    state.lastVideoTime = video.currentTime;
    try {
      await processFrame();
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      state.running = false;
      break;
    }
  }
}

function stopSource() {
  state.generation += 1;
  state.running = false;
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  video.pause();
  video.srcObject = null;
  video.removeAttribute("src");
  video.load();
}

async function startLoop() {
  $("hint").hidden = true;
  resetTracking();
  state.fps = 0;
  state.lastFrameTime = 0;
  state.running = true;
  loop(state.generation);
}

async function startCamera() {
  stopSource();
  const generation = state.generation;
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera needs a secure context (HTTPS or localhost)");
    return;
  }
  const constraints = {
    video: { facingMode: { ideal: state.facing }, width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    if (generation !== state.generation) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    video.srcObject = stream;
    await video.play();
  } catch (error) {
    setStatus(`Camera error: ${error.message}`);
    return;
  }
  $("hint").hidden = true;
  await ensureSession();
  if (generation !== state.generation) return;
  await startLoop();
}

async function startVideo(url) {
  stopSource();
  const generation = state.generation;
  await ensureSession();
  if (generation !== state.generation) return;
  video.src = url;
  video.loop = true;
  video.addEventListener("loadeddata", () => video.play(), { once: true });
  await startLoop();
}

function init() {
  const backends = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
  $("backend").innerHTML = backends.map((name) => `<option value="${name}">${name}</option>`).join("");

  window.addEventListener("error", (event) => setStatus(`Error: ${event.message}`));
  window.addEventListener("unhandledrejection", (event) => setStatus(`Error: ${event.reason?.message || event.reason}`));

  $("camera").onclick = () => startCamera();
  $("flip").onclick = () => {
    state.facing = state.facing === "environment" ? "user" : "environment";
    return startCamera();
  };
  $("demo").onclick = () => startVideo(DEMO_URL);
  $("file").onchange = (event) => {
    const [file] = event.target.files;
    if (file) startVideo(URL.createObjectURL(file));
  };
  $("stop").onclick = () => {
    stopSource();
    $("hint").hidden = false;
  };
  $("threshold").oninput = (event) => {
    $("thresholdValue").textContent = Number(event.target.value).toFixed(2);
  };
  $("backend").onchange = () => ensureSession();

  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
  if (new URLSearchParams(location.search).has("demo")) startVideo(DEMO_URL);
}

init();
