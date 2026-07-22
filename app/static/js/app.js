"use strict";

const state = {
  cameras: [], videos: [], videoStatus: {}, races: [], race: null, laps: [], resultGroups: [], camera: {}, socket: null,
  view: "setup", finishLine: MotoFinishLine.createState(),
  clock: MotoRaceClock.createState(), refreshing: false,
  cameraApplying: false, shuttingDown: false, lapRevision: 0, lastCameraRecoveryMs: 0,
};
const preview = {
  generation: 0, selector: null, endpoint: null, timer: null, controller: null,
  objectUrl: null, active: false, streaming: false, media: false, failures: 0,
  lastSequence: -1,
};
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function acceptServerFinishLine(line, force = false) {
  if (!line) return false;
  if (force || !state.finishLine.initialized) {
    MotoFinishLine.initialize(state.finishLine, line);
    return true;
  }
  return MotoFinishLine.applyRemote(state.finishLine, line);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `Ошибка HTTP ${response.status}`;
    try { message = (await response.json()).error?.message || message; } catch { /* not JSON */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function showMessage(text, error = false) {
  const box = $("#message");
  box.textContent = text;
  box.className = `message${error ? " error" : ""}`;
  box.hidden = false;
  window.setTimeout(() => { box.hidden = true; }, 5000);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("motoLaps.theme", theme);
  $("#theme-toggle").textContent = theme === "dark" ? "Светлая тема" : "Тёмная тема";
}

function navigate(view) {
  state.view = view;
  $$(".view").forEach((item) => item.classList.toggle("active", item.id === `view-${view}`));
  $$(".nav").forEach((item) => item.classList.toggle("active", item.dataset.view === view));
  startPreview();
  if (view === "results") loadResults();
  if (view === "camera") refreshNumberBoardCrop();
}

function cameraOptions() {
  return state.cameras.map((camera) => `<option value="${escapeHtml(camera.identifier)}">${escapeHtml(camera.label)}</option>`).join("");
}

function renderCameras() {
  const selected = state.camera.selected_camera || "";
  for (const selector of ["#setup-camera", "#camera-select"]) {
    const element = $(selector);
    element.innerHTML = '<option value="">Выберите камеру</option>' + cameraOptions();
    if ([...element.options].some((option) => option.value === selected)) element.value = selected;
  }
}

function renderVideos() {
  const options = state.videos.map((video) => (
    `<option value="${escapeHtml(video.id)}">${escapeHtml(video.original_name)} · ${video.width}×${video.height} · ${video.duration_s.toFixed(1)} с</option>`
  )).join("");
  for (const selector of ["#setup-video", "#video-select"]) {
    const element = $(selector);
    const previous = element.value;
    element.innerHTML = '<option value="">Выберите загруженное видео</option>' + options;
    if (state.videos.some((video) => video.id === previous)) element.value = previous;
    else if (state.videos[0]) element.value = state.videos[0].id;
  }
}

function updateSourceMode() {
  const uploaded = $("#source-mode").value === "video";
  $("#setup-camera-fields").hidden = uploaded;
  $("#setup-video-fields").hidden = !uploaded;
  startPreview(true);
}

async function uploadVideo() {
  const input = $("#video-file");
  if (!input.files?.length) throw new Error("Выберите видеофайл.");
  const form = new FormData();
  form.append("file", input.files[0]);
  const response = await fetch("/api/videos", { method: "POST", body: form });
  if (!response.ok) {
    let message = `Ошибка HTTP ${response.status}`;
    try { message = (await response.json()).error?.message || message; } catch { /* not JSON */ }
    throw new Error(message);
  }
  const uploaded = await response.json();
  state.videos = await request("/api/videos");
  renderVideos();
  $("#video-select").value = uploaded.id;
  $("#setup-video").value = uploaded.id;
  startPreview(true);
  showMessage(`Видео «${uploaded.original_name}» загружено и проверено.`);
}

async function applyCamera(identifier, force = false) {
  if (!identifier) throw new Error("Выберите камеру.");
  if (!force && state.camera.status === "running"
      && state.camera.selected_camera === identifier
      && state.camera.active_source === identifier) {
    startPreview();
    return;
  }
  if (state.cameraApplying) throw new Error("Камера уже подключается.");
  state.cameraApplying = true;
  for (const control of [$("#setup-camera"), $("#camera-select"), $("#apply-camera")]) control.disabled = true;
  try {
    state.camera = await request("/api/camera", {
      method: "PUT", body: JSON.stringify({ camera_identifier: identifier }),
    });
    acceptServerFinishLine(state.camera.finish_line);
    $("#setup-camera").value = identifier;
    $("#camera-select").value = identifier;
    startPreview(true);
    renderVisionOverlays();
  } finally {
    state.cameraApplying = false;
    for (const control of [$("#setup-camera"), $("#camera-select"), $("#apply-camera")]) control.disabled = false;
  }
}

function stopPreview(clearImages = true) {
  preview.generation += 1;
  preview.active = false;
  preview.selector = null;
  preview.endpoint = null;
  if (preview.timer !== null) window.clearTimeout(preview.timer);
  if (preview.controller) preview.controller.abort();
  if (preview.objectUrl) URL.revokeObjectURL(preview.objectUrl);
  Object.assign(preview, {
    timer: null, controller: null, objectUrl: null, streaming: false, media: false,
    lastSequence: -1,
  });
  for (const image of [$("#setup-preview"), $("#camera-preview"), $("#active-preview"), $("#video-preview")]) {
    image.onerror = null;
  }
  for (const player of [$("#active-video-player"), $("#video-player")]) {
    player.pause();
    player.hidden = true;
    if (clearImages) {
      player.removeAttribute("src");
      player.load();
    }
  }
  if (clearImages) {
    for (const image of [$("#setup-preview"), $("#camera-preview"), $("#active-preview"), $("#video-preview")]) image.removeAttribute("src");
  }
}

function syncVideoPlayer(player) {
  const positionMs = Number(state.videoStatus.video_position_ms);
  if (!player || !Number.isFinite(positionMs) || !Number.isFinite(player.duration)) return;
  const running = state.videoStatus.state === "processing" && state.race?.status === "running";
  const playbackPlan = MotoVideoPlayback.plan({
    targetSeconds: positionMs / 1000,
    currentSeconds: player.currentTime,
    rate: state.clock.rate,
    running,
  });
  if (playbackPlan.seekTo !== null) player.currentTime = playbackPlan.seekTo;
  player.playbackRate = playbackPlan.playbackRate;
  if (playbackPlan.shouldPlay) player.play().catch(() => { /* retry after next status */ });
  else {
    player.pause();
    if (Math.abs(player.currentTime - positionMs / 1000) > 0.04) {
      player.currentTime = positionMs / 1000;
    }
  }
}

function startVideoPlayer(generation, player) {
  player.hidden = false;
  const connect = () => {
    if (!preview.active || generation !== preview.generation || !preview.media) return;
    if (player.getAttribute("src") !== preview.endpoint) {
      player.src = preview.endpoint;
      player.load();
    }
    syncVideoPlayer(player);
    window.requestAnimationFrame(renderVisionOverlays);
  };
  player.onloadedmetadata = connect;
  player.onerror = () => {
    if (!preview.active || generation !== preview.generation || !preview.media) return;
    preview.timer = window.setTimeout(connect, 500);
  };
  connect();
}

function startMjpegPreview(generation, image) {
  const connect = () => {
    if (!preview.active || generation !== preview.generation || !preview.streaming) return;
    image.src = `${preview.endpoint}?t=${Date.now()}`;
    window.requestAnimationFrame(renderVisionOverlays);
  };
  image.onerror = () => {
    if (!preview.active || generation !== preview.generation || !preview.streaming) return;
    image.removeAttribute("src");
    preview.timer = window.setTimeout(connect, 250);
  };
  connect();
}

async function loadPreviewFrame(generation, image) {
  if (!preview.active || generation !== preview.generation) return;
  const controller = new AbortController();
  preview.controller = controller;
  let nextUrl = null;
  const liveSnapshot = preview.endpoint === "/api/camera/snapshot";
  let retryDelay = liveSnapshot ? 0 : (preview.endpoint?.includes("/snapshot") ? 1000 : 25);
  try {
    const query = liveSnapshot
      ? `after=${encodeURIComponent(preview.lastSequence)}&t=${Date.now()}`
      : `t=${Date.now()}`;
    const response = await fetch(`${preview.endpoint}?${query}`, {
      cache: "no-store", signal: controller.signal, headers: { Accept: "image/jpeg" },
    });
    if (!response.ok) throw new Error(`Camera frame HTTP ${response.status}`);
    const frame = await response.blob();
    if (!frame.size) throw new Error("Empty camera frame");
    nextUrl = URL.createObjectURL(frame);
    const decoded = new Image();
    decoded.src = nextUrl;
    await decoded.decode();
    if (!preview.active || generation !== preview.generation) return;
    const previousUrl = preview.objectUrl;
    image.src = nextUrl;
    preview.objectUrl = nextUrl;
    if (liveSnapshot) {
      const sequence = Number(response.headers.get("X-Frame-Sequence"));
      if (Number.isFinite(sequence)) preview.lastSequence = sequence;
    }
    nextUrl = null;
    preview.failures = 0;
    if (previousUrl) window.setTimeout(() => URL.revokeObjectURL(previousUrl), 0);
    window.requestAnimationFrame(renderVisionOverlays);
  } catch (error) {
    if (error.name === "AbortError") return;
    preview.failures += 1;
    retryDelay = Math.min(1000, 120 + preview.failures * 80);
  } finally {
    if (nextUrl) URL.revokeObjectURL(nextUrl);
    if (preview.controller === controller) preview.controller = null;
  }
  if (preview.active && generation === preview.generation) {
    preview.timer = window.setTimeout(() => loadPreviewFrame(generation, image), retryDelay);
  }
}

function startPreview(force = false) {
  if (document.hidden) {
    stopPreview();
    return;
  }
  const targets = { setup: "#setup-preview", camera: "#camera-preview", active: "#active-preview", video: "#video-preview" };
  let activeSelector = targets[state.view];
  const selectedVideo = $("#video-select")?.value || $("#setup-video")?.value;
  const videoRace = state.race?.camera_identifier?.startsWith("video:");
  const wantsVideo = state.view === "video"
    || (state.view === "active" && videoRace)
    || (state.view === "setup" && $("#source-mode")?.value === "video");
  const videoId = videoRace ? state.race.camera_identifier.slice(6) : selectedVideo;
  const activeVideoMatches = state.videoStatus.video?.id === videoId;
  let endpoint = wantsVideo
    ? (["processing", "paused", "completed"].includes(state.videoStatus.state) && activeVideoMatches
      ? "/api/video/frame"
      : (videoId ? `/api/videos/${videoId}/snapshot` : null))
    : "/api/camera/snapshot";
  const media = wantsVideo && activeVideoMatches
    && ["processing", "paused", "completed"].includes(state.videoStatus.state)
    && ["active", "video"].includes(state.view);
  if (media) {
    activeSelector = state.view === "active" ? "#active-video-player" : "#video-player";
    endpoint = `/api/videos/${videoId}/media`;
  }
  const sourceReady = wantsVideo ? Boolean(endpoint) : state.camera.status === "running";
  if (!sourceReady || !activeSelector) {
    stopPreview();
    window.requestAnimationFrame(renderVisionOverlays);
    return;
  }
  if (!force && preview.active && preview.selector === activeSelector && preview.endpoint === endpoint) return;
  preview.generation += 1;
  if (preview.timer !== null) window.clearTimeout(preview.timer);
  if (preview.controller) preview.controller.abort();
  if (preview.objectUrl) URL.revokeObjectURL(preview.objectUrl);
  preview.objectUrl = null;
  for (const image of [$("#setup-preview"), $("#camera-preview"), $("#active-preview"), $("#video-preview")]) {
    image.onerror = null;
    image.hidden = media;
    if (!image.matches(activeSelector)) image.removeAttribute("src");
  }
  for (const player of [$("#active-video-player"), $("#video-player")]) {
    if (!player.matches(activeSelector)) {
      player.pause();
      player.hidden = true;
      player.removeAttribute("src");
      player.load();
    }
  }
  const streaming = endpoint === "/api/camera/frame";
  Object.assign(preview, {
    active: true, selector: activeSelector, endpoint, streaming, media,
    failures: 0, timer: null, controller: null, lastSequence: -1,
  });
  if (media) startVideoPlayer(preview.generation, $(activeSelector));
  else if (streaming) startMjpegPreview(preview.generation, $(activeSelector));
  else loadPreviewFrame(preview.generation, $(activeSelector));
  window.requestAnimationFrame(renderVisionOverlays);
}

function displayedImageRect(video, image) {
  const sourceWidth = image.videoWidth || image.naturalWidth || video.clientWidth || 1;
  const sourceHeight = image.videoHeight || image.naturalHeight || video.clientHeight || 1;
  return MotoFinishLine.containedImageRect(
    video.clientWidth || 1,
    video.clientHeight || 1,
    sourceWidth,
    sourceHeight,
  );
}

function renderVisionOverlays() {
  const editableLine = state.finishLine.editable;
  for (const overlay of $$(".vision-overlay")) {
    const video = overlay.closest(".video");
    const image = video?.querySelector("video:not([hidden]), img:not([hidden])");
    if (!video || !image) continue;
    const rect = displayedImageRect(video, image);
    Object.assign(overlay.style, {
      left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px`,
    });
    overlay.setAttribute("viewBox", "0 0 1000 1000");
    overlay.setAttribute("preserveAspectRatio", "none");
    const line = overlay.querySelector(".finish-line");
    if (line) {
      line.setAttribute("x1", editableLine.x1 * 1000); line.setAttribute("y1", editableLine.y1 * 1000);
      line.setAttribute("x2", editableLine.x2 * 1000); line.setAttribute("y2", editableLine.y2 * 1000);
    }
    const start = overlay.querySelector('[data-endpoint="start"]');
    const end = overlay.querySelector('[data-endpoint="end"]');
    if (start) { start.setAttribute("cx", editableLine.x1 * 1000); start.setAttribute("cy", editableLine.y1 * 1000); }
    if (end) { end.setAttribute("cx", editableLine.x2 * 1000); end.setAttribute("cy", editableLine.y2 * 1000); }
    const layer = overlay.querySelector(".detection-layer");
    if (layer) {
      const videoSource = ["video-preview", "video-player", "active-video-player"].includes(image.id)
        || (image.id === "active-preview" && state.race?.camera_identifier?.startsWith("video:"));
      const runtime = videoSource ? state.videoStatus : state.camera;
      const groups = [];
      for (const track of runtime.tracks || []) {
        const label = track.racing_number ? `№ ${escapeHtml(track.racing_number)}` : `Motorcycle #${track.track_id}`;
        groups.push(overlayBox(track, "tracked-object", label));
      }
      for (const board of runtime.boards || []) groups.push(overlayBox(board, "board-object", "Number board"));
      for (const digits of runtime.digit_regions || []) {
        groups.push(overlayBox(digits, "digit-object", digits.text ? `Digits ${escapeHtml(digits.text)}` : "Digits"));
      }
      layer.innerHTML = groups.join("");
    }
  }
  renderRecognitionPanels();
}

function flashFinishLineCrossing() {
  for (const line of $$(".vision-overlay .finish-line")) {
    line.classList.remove("crossed");
    // Restart the short pulse even when two motorcycles cross close together.
    void line.getBoundingClientRect();
    line.classList.add("crossed");
    window.setTimeout(() => line.classList.remove("crossed"), 850);
  }
}

function overlayBox(item, cssClass, label) {
  const x = Number(item.x1) * 1000; const y = Number(item.y1) * 1000;
  const width = (Number(item.x2) - Number(item.x1)) * 1000;
  const height = (Number(item.y2) - Number(item.y1)) * 1000;
  return `<g class="${cssClass}"><rect x="${x}" y="${y}" width="${width}" height="${height}"></rect><text x="${x + 6}" y="${Math.max(24, y - 8)}">${label}</text></g>`;
}

function renderRecognitionPanels() {
  renderRecognitionPanel(state.camera, "#stable-number", "#stable-state", "#recognition-history");
  renderRecognitionPanel(state.videoStatus, "#video-stable-number", "#video-stable-state", "#video-recognition-history");
}

function renderRecognitionPanel(runtime, numberSelector, stateSelector, historySelector) {
  const view = MotoRecognitionView.build(runtime);
  $(numberSelector).textContent = view.number;
  $(stateSelector).textContent = view.state;
  $(historySelector).innerHTML = view.history.map((item) => (
    `<li><strong>${escapeHtml(item.racing_number)}</strong> — ${item.crossed ? `пересечение, круг ${item.lap_number}` : "распознан"} · ${new Date(item.timestamp).toLocaleTimeString("ru-RU")}</li>`
  )).join("");
}

async function saveFinishLine() {
  const token = MotoFinishLine.beginSave(state.finishLine);
  try {
    const response = await request("/api/camera/line", {
      method: "PUT", body: JSON.stringify(token.line),
    });
    state.camera = { ...state.camera, ...response };
    if (MotoFinishLine.completeSave(state.finishLine, token, response.finish_line)) {
      renderVisionOverlays();
      showMessage("Финишная линия сохранена.");
    }
  } catch (error) {
    MotoFinishLine.failSave(state.finishLine, token);
    renderVisionOverlays();
    throw new Error(`Линия осталась на новом месте, но сохранить её не удалось. Повторите сохранение. ${error.message}`);
  }
}

function installLineEditor(selector) {
  const overlay = $(selector);
  if (!overlay) return;
  let drag = null;
  const position = (event) => {
    const rect = overlay.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width))),
      y: Math.max(0, Math.min(1, (event.clientY - rect.top) / Math.max(1, rect.height))),
    };
  };
  overlay.addEventListener("pointerdown", (event) => {
    if (drag || state.finishLine.dragging) return;
    const startPoint = position(event);
    const rect = overlay.getBoundingClientRect();
    const mode = MotoFinishLine.hitTest(
      state.finishLine.editable, startPoint, Math.max(1, rect.width), Math.max(1, rect.height),
    );
    if (!mode) return;
    drag = {
      mode, pointerId: event.pointerId, startPoint,
      startLine: MotoFinishLine.cloneLine(state.finishLine.editable),
    };
    state.finishLine.dragging = true;
    overlay.classList.add("dragging");
    overlay.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  overlay.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const point = position(event);
    const next = drag.mode === "segment"
      ? MotoFinishLine.translateLine(
        drag.startLine, point.x - drag.startPoint.x, point.y - drag.startPoint.y,
      )
      : MotoFinishLine.moveEndpoint(drag.startLine, drag.mode, point);
    MotoFinishLine.setEditable(state.finishLine, next);
    renderVisionOverlays();
    event.preventDefault();
  });
  const finish = async (event) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (overlay.hasPointerCapture(event.pointerId)) overlay.releasePointerCapture(event.pointerId);
    drag = null;
    state.finishLine.dragging = false;
    overlay.classList.remove("dragging");
    if (!state.finishLine.dirty) return;
    try { await saveFinishLine(); } catch (error) { showMessage(error.message, true); }
  };
  overlay.addEventListener("pointerup", finish);
  overlay.addEventListener("pointercancel", finish);
}

async function refreshNumberBoardCrop() {
  if (state.view !== "camera" || !$(".debug-crop")?.open) return;
  try {
    const response = await fetch(`/api/camera/number-board?t=${Date.now()}`, { cache: "no-store" });
    if (response.status === 204 || !response.ok) return;
    const image = $("#number-board-preview");
    const oldUrl = image.dataset.objectUrl;
    const nextUrl = URL.createObjectURL(await response.blob());
    image.src = nextUrl;
    image.dataset.objectUrl = nextUrl;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
  } catch { /* crop is optional */ }
}

async function loadBase() {
  try {
    await request("/api/health");
    $("#server-state").textContent = "Сервер подключён";
    [state.cameras, state.camera, state.races, state.videos, state.videoStatus] = await Promise.all([
      request("/api/cameras"), request("/api/camera"), request("/api/races"),
      request("/api/videos"), request("/api/video/status"),
    ]);
    renderCameras(); renderVideos(); updateSourceMode();
    acceptServerFinishLine(state.camera.finish_line, true);
    state.race = state.races.find((race) => ["running", "paused"].includes(race.status)) || state.races[0] || null;
    renderCameras(); renderRace(); startPreview(); renderVisionOverlays(); connectSocket();
    // Camera startup must never block loading race controls and results. A
    // saved source is restored in the background; an explicit selection still
    // awaits the same operation and reports a structured error.
    if (state.camera.status !== "running" && !state.race?.camera_identifier?.startsWith("video:")) {
      const previewStart = state.camera.selected_camera
        ? applyCamera(state.camera.selected_camera)
        : request("/api/camera/demo", { method: "POST" }).then(() => request("/api/camera"));
      previewStart.then((camera) => {
        if (camera) state.camera = camera;
        acceptServerFinishLine(state.camera.finish_line);
        renderCameras(); startPreview(true); renderVisionOverlays();
      }).catch(async (cameraError) => {
        showMessage(cameraError.message, true);
        state.camera = await request("/api/camera");
        renderCameras(); startPreview();
      });
    }
  } catch (error) {
    $("#server-state").textContent = "Сервер недоступен";
    showMessage(error.message, true);
  }
}

async function refreshCameras() {
  const button = $("#refresh-cameras");
  button.disabled = true;
  try {
    state.cameras = await request("/api/cameras");
    renderCameras();
    showMessage(`Найдено камер: ${state.cameras.filter((item) => item.source_type === "webcam").length}.`);
  } finally { button.disabled = false; }
}

async function shutdownApplication() {
  if (!confirm("Завершить Moto Laps? Активная гонка будет поставлена на паузу.")) return;
  state.shuttingDown = true;
  stopPreview();
  if (state.socket) state.socket.close();
  $("#shutdown-app").disabled = true;
  try { await request("/api/system/shutdown", { method: "POST" }); } catch { /* process may exit */ }
  document.body.innerHTML = '<div class="shutdown-screen"><div><h1>Moto Laps остановлен</h1><p>Камера и распознавание выключены. Для запуска дважды щёлкните <strong>Start Moto Laps.bat</strong>.</p></div></div>';
  window.setTimeout(() => window.close(), 500);
}

function renderRace() {
  const race = state.race;
  $("#active-name").textContent = race?.name || "Активная гонка";
  $("#race-status").textContent = statusLabel(race?.status);
  $("#pause-race").disabled = race?.status !== "running";
  $("#resume-race").disabled = race?.status !== "paused";
  $("#finish-race").disabled = !["running", "paused"].includes(race?.status);
  const recording = state.camera?.recording;
  const recordingState = $("#recording-state");
  if (recordingState) {
    recordingState.textContent = race?.camera_identifier?.startsWith("webcam:")
      ? (recording?.active ? "Идёт" : (recording?.error ? "Ошибка" : "Сохранена"))
      : "Не требуется";
  }
  if (race?.elapsed_ns !== null && race?.elapsed_ns !== undefined) {
    const isVideo = Boolean(race.camera_identifier?.startsWith("video:"));
    const videoAdvancing = !state.videoStatus.state || state.videoStatus.state === "processing";
    MotoRaceClock.sync(state.clock, {
      raceId: race.id,
      elapsedNs: Number(race.elapsed_ns),
      nowMs: performance.now(),
      running: race.status === "running" && (!isVideo || videoAdvancing),
      isVideo,
    });
  }
  renderRaceTime();
}

function renderRaceTime() {
  if (state.shuttingDown) return;
  const race = state.race;
  if (!race || race.elapsed_ns === null || race.elapsed_ns === undefined) {
    $("#race-time").textContent = "00:00.000";
    return;
  }
  const elapsed = MotoRaceClock.value(state.clock, performance.now());
  $("#race-time").textContent = formatDuration(elapsed);
}

async function startRace(event) {
  event.preventDefault();
  try {
    const videoMode = $("#source-mode").value === "video";
    const source = videoMode ? `video:${$("#setup-video").value}` : $("#setup-camera").value;
    if (videoMode && source === "video:") throw new Error("Выберите загруженное видео.");
    if (!videoMode) await applyCamera(source);
    const race = await request("/api/races", {
      method: "POST",
      body: JSON.stringify({
        name: $("#race-name").value.trim(),
        description: $("#race-description").value.trim() || null,
        required_laps: Number($("#required-laps").value),
        camera_identifier: source,
      }),
    });
    state.race = await request(`/api/races/${race.id}/start`, { method: "POST" });
    state.races = await request("/api/races");
    state.laps = [];
    state.lapRevision += 1;
    $("#recognized-number").textContent = "—";
    state.videoStatus = videoMode ? await request("/api/video/status") : {};
    renderRecent(); renderRace(); startPreview(true); connectSocket(); navigate("active");
    showMessage("Гонка запущена.");
  } catch (error) { showMessage(error.message, true); }
}

async function transition(action) {
  if (!state.race) return;
  try {
    state.race = await request(`/api/races/${state.race.id}/${action}`, { method: "POST" });
    if (state.race.camera_identifier?.startsWith("video:")) {
      state.videoStatus = await request("/api/video/status");
      startPreview(true);
    }
    renderRace();
    if (action === "finish") { await loadResults(); navigate("results"); }
  } catch (error) { showMessage(error.message, true); }
}

async function refreshActive() {
  if (document.hidden || state.shuttingDown || !state.race || state.refreshing) return;
  if (!["running", "paused"].includes(state.race.status) && state.view !== "results") return;
  state.refreshing = true;
  try {
    const raceId = state.race.id;
    const revisionAtRequest = state.lapRevision;
    const isVideo = state.race.camera_identifier?.startsWith("video:");
    const [race, camera, laps, videoStatus] = await Promise.all([
      request(`/api/races/${raceId}`), request("/api/camera"),
      request(`/api/races/${raceId}/laps?sort_by=recorded&direction=desc`),
      isVideo ? request("/api/video/status") : Promise.resolve({}),
    ]);
    if (state.race?.id !== raceId) return;
    state.race = race;
    state.camera = camera;
    if (isVideo) state.videoStatus = videoStatus;
    acceptServerFinishLine(camera.finish_line);
    $("#recognized-number").textContent = (isVideo ? videoStatus : camera).recognized_number || "—";
    state.laps = revisionAtRequest === state.lapRevision ? laps : MotoLapState.mergeLapRows(state.laps, laps);
    renderRace(); renderRecent(); renderVisionOverlays();
  } catch { /* next poll retries */ }
  finally { state.refreshing = false; }
}

function renderRecent() {
  const rows = state.laps.slice(0, 12);
  $("#recent-laps").innerHTML = rows.map((lap) => `<tr><td><strong>${escapeHtml(lap.racing_number)}</strong></td><td>${lap.lap_number}</td><td>${formatDuration(lap.lap_time_ns)}</td></tr>`).join("");
  $("#recent-empty").hidden = rows.length > 0;
}

async function loadResults() {
  try {
    const sort = $("#sort-laps").value;
    state.races = await request("/api/races");
    state.resultGroups = await Promise.all(state.races.map(async (race) => {
      const [laps, results, recordings] = await Promise.all([
        request(`/api/races/${race.id}/laps?sort_by=${sort}`),
        request(`/api/races/${race.id}/results`),
        request(`/api/races/${race.id}/recordings`),
      ]);
      return { race: results.race, laps, summary: results.summary, recordings };
    }));
    renderResults();
  } catch (error) { showMessage(error.message, true); }
}

function renderResults() {
  const container = $("#race-results");
  $("#results-empty").hidden = state.resultGroups.length > 0;
  container.innerHTML = state.resultGroups.map(({ race, laps, summary, recordings }) => `
    <section class="result-race card" data-race-id="${race.id}">
      <div class="result-race-heading"><div><h2>${escapeHtml(race.name)}</h2><p>${statusLabel(race.status)} · ${race.required_laps} круг(а)${race.description ? ` · ${escapeHtml(race.description)}` : ""}</p></div><button class="export-race primary" data-race-id="${race.id}">Экспорт Excel</button></div>
      <div class="summary">${summary.map((item) => `<article><span>Номер ${escapeHtml(item.racing_number)}</span><strong>${item.completed_laps}/${race.required_laps} кругов</strong><small>${formatDuration(item.total_time_ns)} · ${item.finished ? "финишировал" : "в гонке"}</small></article>`).join("")}</div>
      ${(recordings || []).length ? `<p class="recording-links"><strong>Записи камеры:</strong> ${(recordings || []).map((item, index) => `<a href="${item.download_url}" download="${escapeHtml(item.filename)}">часть ${index + 1}</a>`).join(" · ")}</p>` : ""}
      <div class="table-wrap"><table><thead><tr><th>Номер мотоцикла</th><th>Круг</th><th>Время круга</th><th>Время гонки</th><th>Записан</th><th>Финиш</th><th></th></tr></thead><tbody>${laps.map((lap) => `<tr><td><strong>${escapeHtml(lap.racing_number)}</strong></td><td>${lap.lap_number}</td><td>${formatDuration(lap.lap_time_ns)}</td><td>${formatDuration(lap.race_elapsed_ns)}</td><td>${new Date(lap.detected_at_utc).toLocaleTimeString("ru-RU")}</td><td class="${lap.finished ? "finished" : ""}">${lap.finished ? "Финиш" : "—"}</td><td><button class="edit-lap" data-race-id="${race.id}" data-id="${lap.id}">Исправить</button></td></tr>`).join("")}</tbody></table>${laps.length ? "" : '<p class="empty">Записанных кругов нет.</p>'}</div>
    </section>`).join("");
}

function openCorrection(raceId, id) {
  const group = state.resultGroups.find((item) => String(item.race.id) === String(raceId));
  const lap = group?.laps.find((item) => String(item.id) === String(id));
  if (!lap) return;
  $("#correction-race-id").value = raceId;
  $("#correction-id").value = lap.id;
  $("#correction-number").value = lap.racing_number;
  $("#correction-lap").value = lap.lap_number;
  $("#correction-dialog").showModal();
}

async function saveCorrection(event) {
  event.preventDefault();
  try {
    const id = $("#correction-id").value;
    const raceId = $("#correction-race-id").value;
    await request(`/api/races/${raceId}/laps/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ racing_number: $("#correction-number").value.trim(), lap_number: Number($("#correction-lap").value) }),
    });
    $("#correction-dialog").close(); await loadResults();
  } catch (error) { showMessage(error.message, true); }
}

async function deleteLap() {
  if (!confirm("Удалить эту запись круга?")) return;
  try {
    const raceId = $("#correction-race-id").value;
    await request(`/api/races/${raceId}/laps/${$("#correction-id").value}`, { method: "DELETE" });
    $("#correction-dialog").close(); await loadResults();
  } catch (error) { showMessage(error.message, true); }
}

async function exportRace(raceId) {
  const response = await fetch(`/api/races/${raceId}/export`);
  if (!response.ok) { showMessage("Не удалось создать Excel.", true); return; }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob); const link = document.createElement("a");
  link.href = url; link.download = `race-${raceId}.xlsx`; link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function connectSocket() {
  if (state.socket) state.socket.close();
  if (!state.race) return;
  state.socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`);
  state.socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "lap_recorded" && message.data?.race_id === state.race?.id) {
        const lap = { ...message.data.lap, finished: message.data.lap.lap_number >= state.race.required_laps };
        state.laps = [lap, ...state.laps.filter((item) => item.id !== lap.id)];
        state.lapRevision += 1;
        $("#recognized-number").textContent = lap.racing_number;
        flashFinishLineCrossing();
        renderRecent();
        return;
      }
    } catch { /* authoritative polling remains active */ }
    refreshActive();
  };
}

async function refreshCameraStatus() {
  if (document.hidden || state.shuttingDown || state.refreshing) return;
  try {
    state.camera = await request("/api/camera");
    acceptServerFinishLine(state.camera.finish_line);
    renderVisionOverlays(); startPreview(); refreshNumberBoardCrop();
    const frameAge = state.camera.frame_age_ms;
    const stale = state.camera.status === "running"
      && (frameAge === null || frameAge === undefined || Number(frameAge) > 1500);
    const now = Date.now();
    if (stale && state.camera.selected_camera && !state.cameraApplying
        && now - state.lastCameraRecoveryMs > 5000) {
      state.lastCameraRecoveryMs = now;
      await applyCamera(state.camera.selected_camera, true);
    }
  } catch { /* next poll retries */ }
}

async function refreshVideoStatus() {
  const relevant = state.view === "video" || state.race?.camera_identifier?.startsWith("video:");
  if (document.hidden || state.shuttingDown || !relevant) return;
  try {
    state.videoStatus = await request("/api/video/status");
    const videoRace = state.race?.camera_identifier?.startsWith("video:");
    const positionMs = Number(state.videoStatus.video_position_ms);
    if (videoRace && Number.isFinite(positionMs) && state.videoStatus.race_id === state.race.id) {
      MotoRaceClock.sync(state.clock, {
        raceId: state.race.id,
        elapsedNs: positionMs * 1e6,
        nowMs: performance.now(),
        running: state.race.status === "running" && state.videoStatus.state === "processing",
        isVideo: true,
      });
      renderRaceTime();
    }
    for (const player of [$("#active-video-player"), $("#video-player")]) {
      if (!player.hidden) syncVideoPlayer(player);
    }
    const scanning = state.videoStatus.phase === "scanning";
    const processed = scanning
      ? state.videoStatus.scan_processed_frames || 0
      : state.videoStatus.processed_frames || 0;
    const total = state.videoStatus.total_frames || 0;
    const phase = scanning ? "быстрый поиск мотоциклов" : "точный анализ проездов";
    $("#video-progress").textContent = total
      ? `Этап: ${phase}. Обработано ${processed} из ${total} кадров (${Math.round((state.videoStatus.progress || 0) * 100)}%).`
      : "Видео ещё не запущено как источник гонки.";
    startPreview(); renderVisionOverlays();
  } catch { /* next poll retries */ }
}

function statusLabel(value) {
  return ({ draft: "Черновик", running: "Идёт", paused: "Пауза", finished: "Завершена" })[value] || "—";
}
function formatDuration(ns) {
  const ms = Math.max(0, Math.floor(Number(ns || 0) / 1e6));
  const minutes = Math.floor(ms / 60000); const seconds = Math.floor(ms % 60000 / 1000);
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

$$(".nav").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.view)));
$("#race-form").addEventListener("submit", startRace);
$("#setup-camera").addEventListener("change", async (event) => {
  try { await applyCamera(event.target.value); } catch (error) { showMessage(error.message, true); }
});
$("#apply-camera").addEventListener("click", async () => {
  try { await applyCamera($("#camera-select").value); showMessage("Камера сохранена."); } catch (error) { showMessage(error.message, true); }
});
$("#refresh-cameras").addEventListener("click", refreshCameras);
$("#source-mode").addEventListener("change", updateSourceMode);
$("#upload-video").addEventListener("click", async () => {
  try { await uploadVideo(); } catch (error) { showMessage(error.message, true); }
});
$("#video-select").addEventListener("change", (event) => {
  $("#setup-video").value = event.target.value;
  startPreview(true);
});
$("#setup-video").addEventListener("change", (event) => {
  $("#video-select").value = event.target.value;
  startPreview(true);
});
$("#restart-video").addEventListener("click", async () => {
  try {
    state.videoStatus = await request("/api/video/restart", { method: "POST" });
    startPreview(true);
  } catch (error) { showMessage(error.message, true); }
});
$("#video-save-line").addEventListener("click", async () => {
  try { await saveFinishLine(); } catch (error) { showMessage(error.message, true); }
});
$("#shutdown-app").addEventListener("click", shutdownApplication);
$("#theme-toggle").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
$("#save-line").addEventListener("click", async () => {
  try { await saveFinishLine(); } catch (error) { showMessage(error.message, true); }
});
$("#reset-line").addEventListener("click", async () => {
  MotoFinishLine.setEditable(state.finishLine, MotoFinishLine.DEFAULT_LINE);
  renderVisionOverlays();
  try { await saveFinishLine(); } catch (error) { showMessage(error.message, true); }
});
$("#pause-race").addEventListener("click", () => transition("pause"));
$("#resume-race").addEventListener("click", () => transition("resume"));
$("#finish-race").addEventListener("click", () => transition("finish"));
$("#sort-laps").addEventListener("change", loadResults);
$("#race-results").addEventListener("click", (event) => {
  const edit = event.target.closest(".edit-lap"); if (edit) openCorrection(edit.dataset.raceId, edit.dataset.id);
  const exportButton = event.target.closest(".export-race"); if (exportButton) exportRace(exportButton.dataset.raceId);
});
$("#correction-form").addEventListener("submit", saveCorrection);
$("#delete-lap").addEventListener("click", deleteLap);
for (const image of [$("#setup-preview"), $("#camera-preview"), $("#active-preview"), $("#video-preview")]) image.addEventListener("load", renderVisionOverlays);
for (const player of [$("#active-video-player"), $("#video-player")]) {
  player.addEventListener("loadedmetadata", renderVisionOverlays);
  player.addEventListener("timeupdate", renderVisionOverlays);
}
window.addEventListener("resize", renderVisionOverlays);
window.addEventListener("beforeunload", () => stopPreview());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopPreview();
    return;
  }
  refreshCameraStatus();
  refreshVideoStatus();
  refreshActive();
  startPreview(true);
});
installLineEditor("#editable-line");
installLineEditor("#video-editable-line");
setTheme(document.documentElement.dataset.theme || "dark");
loadBase();
window.setInterval(refreshActive, 1000);
window.setInterval(refreshCameraStatus, 2000);
window.setInterval(refreshVideoStatus, 500);
function animateRaceClock() {
  renderRaceTime();
  window.requestAnimationFrame(animateRaceClock);
}
window.requestAnimationFrame(animateRaceClock);
