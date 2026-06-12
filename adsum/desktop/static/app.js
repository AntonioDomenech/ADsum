const state = {
  devices: { microphones: [], speakers: [] },
  poll: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, body = null) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open(body ? "POST" : "GET", path, true);
    request.setRequestHeader("Accept", "application/json");
    if (body) {
      request.setRequestHeader("Content-Type", "application/json");
    }
    request.onload = () => {
      let payload = {};
      try {
        payload = request.responseText ? JSON.parse(request.responseText) : {};
      } catch (error) {
        reject(new Error("Invalid response from ADsum."));
        return;
      }
      if (request.status < 200 || request.status >= 300 || payload.error) {
        reject(new Error(payload.error || "Request failed."));
        return;
      }
      resolve(payload);
    };
    request.onerror = () => reject(new Error("ADsum local server is unreachable."));
    request.send(body ? JSON.stringify(body) : null);
  });
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 4600);
}

function fillSelect(select, devices, fallback) {
  select.innerHTML = "";
  if (!devices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = fallback;
    select.appendChild(option);
    return;
  }
  for (const device of devices) {
    const option = document.createElement("option");
    option.value = device.id;
    option.textContent = `${device.name}${device.is_default ? " (default)" : ""}`;
    option.dataset.warning = device.warning || "";
    select.appendChild(option);
  }
  const defaultOption = Array.from(select.options).find((option) =>
    option.textContent.includes("(default)")
  );
  if (defaultOption) {
    select.value = defaultOption.value;
  }
}

function updateWarnings() {
  const micOption = $("micSelect").selectedOptions[0];
  const speakerOption = $("speakerSelect").selectedOptions[0];
  $("micWarning").textContent = micOption?.dataset.warning || "";
  $("speakerWarning").textContent = speakerOption?.dataset.warning || "";
}

async function refreshDevices() {
  const payload = await api("/api/devices");
  state.devices = payload.devices;
  fillSelect($("micSelect"), state.devices.microphones, "No microphone found");
  fillSelect($("speakerSelect"), state.devices.speakers, "No system output found");
  updateWarnings();
}

function setBusy(isBusy) {
  $("refreshBtn").disabled = isBusy;
  $("testBtn").disabled = isBusy;
  $("recordBtn").disabled = isBusy;
  $("saveKeyBtn").disabled = isBusy;
  $("transcribeBtn").disabled = isBusy;
}

function selectedPayload() {
  return {
    name: $("sessionName").value,
    microphone_id: $("micSelect").value,
    speaker_id: $("speakerSelect").value,
  };
}

async function startRecording() {
  setBusy(true);
  try {
    await api("/api/start", selectedPayload());
    await refreshStatus();
  } finally {
    setBusy(false);
  }
}

async function stopRecording() {
  $("stopBtn").disabled = true;
  const payload = await api("/api/stop", {});
  renderResult(payload.result);
  await refreshStatus();
}

async function runTest() {
  setBusy(true);
  $("resultBox").textContent = "Running 6 second device test...";
  try {
    const payload = await api("/api/test", {
      ...selectedPayload(),
      duration_seconds: 6,
    });
    renderResult(payload.result);
    showToast("Device test finished.");
    await refreshStatus();
  } finally {
    setBusy(false);
  }
}

async function saveKey() {
  const key = $("apiKey").value.trim();
  if (!key) {
    showToast("Paste a key before saving.");
    return;
  }
  await api("/api/settings/openai-key", { key });
  $("apiKey").value = "";
  $("keyState").textContent = "OpenAI key configured";
  $("keyState").classList.add("ready");
  showToast("OpenAI key saved locally.");
}

async function transcribe() {
  $("transcribeBtn").disabled = true;
  $("transcriptState").textContent = "Transcribing";
  $("transcriptBox").textContent = "Waiting for OpenAI...";
  try {
    const payload = await api("/api/transcribe", {});
    $("transcriptBox").textContent = payload.transcript.text || "(No text returned.)";
    $("transcriptState").textContent = "Done";
  } catch (error) {
    $("transcriptState").textContent = "Error";
    $("transcriptBox").textContent = error.message;
    throw error;
  } finally {
    $("transcribeBtn").disabled = false;
  }
}

function renderResult(result) {
  if (!result) {
    $("resultBox").textContent = "No recording yet.";
    $("transcribeBtn").disabled = true;
    return;
  }
  const metrics = result.metrics || {};
  const lines = [
    `${result.name}`,
    `${result.duration_seconds}s`,
    "",
    `Mic: ${formatMetric(metrics.microphone)}`,
    `System: ${formatMetric(metrics.system)}`,
    `Mixed: ${formatMetric(metrics.mixed)}`,
    "",
    `Folder: ${result.paths.session_dir}`,
  ];
  $("resultBox").textContent = lines.join("\n");
  $("transcribeBtn").disabled = !(result.paths && result.paths.mixed_path);
}

function formatMetric(metric) {
  if (!metric || !metric.path) {
    return "missing";
  }
  return `${metric.duration_seconds}s, peak ${metric.peak}, rms ${metric.rms}`;
}

function renderStatus(status) {
  const recording = status.recording;
  const isRecording = recording.state === "recording";
  $("statePill").textContent = isRecording ? "Recording" : "Idle";
  $("statePill").classList.toggle("recording", isRecording);
  $("recordBtn").disabled = isRecording;
  $("stopBtn").disabled = !isRecording;
  $("testBtn").disabled = isRecording;
  $("refreshBtn").disabled = isRecording;
  $("elapsedText").textContent = secondsToClock(recording.elapsed_seconds || 0);
  const levels = recording.levels || {};
  setMeter("micMeter", levels.microphone || 0);
  setMeter("systemMeter", levels.system || 0);
  if (!isRecording && recording.last_result) {
    renderResult(recording.last_result);
  }
  $("keyState").textContent = status.openai_key_configured
    ? "OpenAI key configured"
    : "OpenAI key not configured";
  $("keyState").classList.toggle("ready", status.openai_key_configured);
  if (status.last_transcript) {
    $("transcriptBox").textContent = status.last_transcript.text || "(No text returned.)";
    $("transcriptState").textContent = "Done";
  }
}

async function refreshStatus() {
  const status = await api("/api/status");
  renderStatus(status);
}

function setMeter(id, rms) {
  const percent = Math.max(0, Math.min(100, Math.sqrt(Math.max(rms, 0)) * 140));
  $(id).style.width = `${percent}%`;
}

function secondsToClock(value) {
  const seconds = Math.max(0, Math.floor(value || 0));
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function wireEvents() {
  $("refreshBtn").addEventListener("click", () => refreshDevices().catch((error) => showToast(error.message)));
  $("recordBtn").addEventListener("click", () => startRecording().catch((error) => showToast(error.message)));
  $("stopBtn").addEventListener("click", () => stopRecording().catch((error) => showToast(error.message)));
  $("testBtn").addEventListener("click", () => runTest().catch((error) => showToast(error.message)));
  $("saveKeyBtn").addEventListener("click", () => saveKey().catch((error) => showToast(error.message)));
  $("transcribeBtn").addEventListener("click", () => transcribe().catch((error) => showToast(error.message)));
  $("micSelect").addEventListener("change", updateWarnings);
  $("speakerSelect").addEventListener("change", updateWarnings);
}

async function init() {
  wireEvents();
  await refreshDevices();
  await refreshStatus();
  state.poll = setInterval(() => refreshStatus().catch(() => {}), 700);
}

init().catch((error) => showToast(error.message));
