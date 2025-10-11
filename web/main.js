// main.js - SPA frontend logic for Ratatat Video Enhancer
// Uses simple placeholders for backend API endpoints.

const API = {
  // Adjust these paths to match your backend routes
  upload: "/api/upload", // multipart/form-data: { file }
  enhance: "/api/enhance", // json: { jobId, model }
  status: "/api/status", // GET with ?jobId=
};

// Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileMeta = document.getElementById("file-meta");
const modelSelect = document.getElementById("model-select");
const enhanceBtn = document.getElementById("enhance-btn");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-label");
const resultCard = document.getElementById("result-card");
const resultVideo = document.getElementById("result-video");
const downloadBtn = document.getElementById("download-btn");
const resetBtn = document.getElementById("reset-btn");

// State
let currentFile = null;
let currentJobId = null;
let polling = null;

function formatFileSize(bytes) {
  if (!bytes && bytes !== 0) return "";
  const units = ["B", "KB", "MB", "GB"]; let i = 0; let n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${units[i]}`;
}

function setProgress(pct, label) {
  const clamped = Math.max(0, Math.min(100, pct|0));
  progressBar.style.width = `${clamped}%`;
  progressLabel.textContent = label ?? `${clamped}%`;
}

function setEnhanceEnabled(enabled) {
  enhanceBtn.disabled = !enabled;
}

function clearResult() {
  resultVideo.removeAttribute("src");
  resultVideo.load();
  downloadBtn.href = "#";
  resultCard.classList.add("hidden");
}

function showResult(url) {
  resultVideo.src = url;
  resultVideo.load();
  resultCard.classList.remove("hidden");
  downloadBtn.href = url;
}

function setDropHighlight(on) {
  dropzone.classList.toggle("drop-highlight", !!on);
}

function showFileMeta(file) {
  fileMeta.textContent = `${file.name} • ${file.type || "video"} • ${formatFileSize(file.size)}`;
}

function validateFile(file) {
  const allowed = ["video/mp4", "video/quicktime", "video/x-msvideo"]; // MP4, MOV, AVI
  if (!allowed.includes(file.type)) {
    throw new Error("Unsupported format. Please use MP4, MOV, or AVI.");
  }
  const max = 100 * 1024 * 1024; // 100MB
  if (file.size > max) {
    throw new Error("File too large. Max 100 MB.");
  }
}

function selectFile(file) {
  try {
    validateFile(file);
    currentFile = file;
    showFileMeta(file);
    setEnhanceEnabled(true);
    clearResult();
  } catch (e) {
    alert(e.message || String(e));
  }
}

// Drag & Drop
['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation(); setDropHighlight(true);
}));
['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation(); setDropHighlight(false);
}));

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('drop', e => {
  const dt = e.dataTransfer;
  if (dt && dt.files && dt.files[0]) {
    selectFile(dt.files[0]);
  }
});

fileInput.addEventListener('change', e => {
  const f = e.target.files?.[0];
  if (f) selectFile(f);
});

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(API.upload, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json(); // expect { jobId, filename }
}

async function startEnhancement(jobId, model) {
  const res = await fetch(API.enhance, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jobId, model })
  });
  if (!res.ok) throw new Error(`Enhance start failed: ${res.status}`);
  return res.json(); // expect { ok: true }
}

async function getStatus(jobId) {
  const url = new URL(API.status, window.location.origin);
  url.searchParams.set('jobId', jobId);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json();
}

async function runEnhancementFlow() {
  if (!currentFile) return;
  try {
    setEnhanceEnabled(false);
    setProgress(3, 'Uploading…');
    const up = await uploadFile(currentFile);
    currentJobId = up.jobId;

    setProgress(5, 'Starting enhancement…');
    await startEnhancement(currentJobId, modelSelect.value);

    // Polling loop
    setProgress(8, 'Processing…');
    polling = setInterval(async () => {
      try {
        const s = await getStatus(currentJobId);
        // Expected: { progress, status, message, resultUrl }
        if (typeof s.progress === 'number') {
          setProgress(s.progress, s.message || `Processing… ${s.progress}%`);
        } else if (s.message) {
          progressLabel.textContent = s.message;
        }
        if (s.status === 'completed' && s.resultUrl) {
          clearInterval(polling); polling = null;
          setProgress(100, 'Done');
          const url = new URL(s.resultUrl, window.location.origin).toString();
          showResult(url);
          setEnhanceEnabled(true);
        } else if (s.status === 'failed') {
          clearInterval(polling); polling = null;
          setProgress(100, 'Failed');
          alert(`Enhancement failed: ${s.message || 'Unknown error'}`);
          setEnhanceEnabled(true);
        }
      } catch (e) {
        clearInterval(polling); polling = null;
        alert(e.message || String(e));
        setEnhanceEnabled(true);
      }
    }, 1200);
  } catch (e) {
    alert(e.message || String(e));
    setEnhanceEnabled(true);
  }
}

enhanceBtn.addEventListener('click', runEnhancementFlow);

resetBtn.addEventListener('click', () => {
  currentFile = null; currentJobId = null;
  fileInput.value = '';
  fileMeta.textContent = '';
  setProgress(0, 'Idle');
  clearResult();
  setEnhanceEnabled(false);
});
