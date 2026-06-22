"use strict";
const $ = (s) => document.querySelector(s);
const SEV_KO = { severe: "심각", moderate: "보통", minor: "경미" };
const HKEY = "rdd_history_v1";

let selectedFile = null;   // File 업로드
let selectedSample = null; // 서버 샘플 파일명
let lastCoco = null;

// ---------- 탭 ----------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  const v = t.dataset.view;
  $("#view-detect").hidden = v !== "detect";
  $("#view-history").hidden = v !== "history";
  if (v === "history") renderHistory();
}));

// ---------- 테마 ----------
const themeBtn = $("#theme-toggle");
function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  themeBtn.textContent = mode === "dark" ? "🌙 다크" : "☀️ 라이트";
  localStorage.setItem("rdd_theme", mode);
}
themeBtn.addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
applyTheme(localStorage.getItem("rdd_theme") || "dark");

// ---------- 이미지 입력 ----------
const dz = $("#dropzone"), fileInput = $("#file-input"), preview = $("#preview");
function showPreview(src) {
  preview.src = src; preview.hidden = false; $("#dropzone-empty").hidden = true;
}
function setFile(file) {
  selectedFile = file; selectedSample = null;
  showPreview(URL.createObjectURL(file));
}
dz.addEventListener("click", () => fileInput.click());
$("#btn-upload").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("drag");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

$("#btn-sample").addEventListener("click", async (e) => {
  e.stopPropagation();
  try {
    const r = await fetch("/api/samples"); const { samples } = await r.json();
    if (!samples || !samples.length) return alert("샘플 이미지가 없습니다. data/images/real 에 이미지를 넣어주세요.");
    selectedSample = samples[Math.floor(Math.random() * samples.length)];
    selectedFile = null;
    showPreview("/samples/" + encodeURIComponent(selectedSample));
  } catch { alert("샘플을 불러오지 못했습니다."); }
});

// ---------- 슬라이더 ----------
const conf = $("#conf");
conf.addEventListener("input", () => { $("#conf-val").textContent = (+conf.value).toFixed(2); });

// ---------- 상태 전환 ----------
function showState(id) {
  ["#state-empty", "#state-loading", "#state-threshold", "#state-error", "#result"]
    .forEach((s) => { $(s).hidden = s !== id; });
}

// ---------- 탐지 실행 ----------
$("#btn-run").addEventListener("click", async () => {
  if (!selectedFile && !selectedSample) { alert("이미지를 먼저 선택하세요."); return; }
  const detector = document.querySelector('input[name="detector"]:checked').value;
  const fd = new FormData();
  fd.append("query", $("#query").value);
  fd.append("detector", detector);
  fd.append("conf", conf.value);
  if (selectedSample) fd.append("sample", selectedSample);
  else fd.append("image", selectedFile);

  showState("#state-loading");
  $("#btn-export").disabled = true;
  try {
    const r = await fetch("/api/detect", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok || data.error) { $("#error-msg").textContent = "오류: " + (data.error || r.status); showState("#state-error"); return; }
    if (!data.detections.length) { showState("#state-threshold"); return; }
    renderResult(data);
    saveHistory(data);
  } catch (err) { $("#error-msg").textContent = "요청 실패: " + err; showState("#state-error"); }
});

$("#btn-reset").addEventListener("click", () => {
  selectedFile = null; selectedSample = null; lastCoco = null;
  preview.hidden = true; $("#dropzone-empty").hidden = false; fileInput.value = "";
  $("#btn-export").disabled = true; showState("#state-empty");
});

// ---------- 결과 렌더 ----------
function renderResult(data) {
  const s = data.stats;
  $("#st-count").textContent = s.count;
  $("#st-severe").textContent = s.severe;
  $("#st-avg").textContent = Math.round(s.avg_conf * 100) + "%";
  $("#st-model").textContent = s.model;

  const d = data.severity_dist, tot = Math.max(1, s.count);
  $("#dist-severe").style.width = (d.severe / tot * 100) + "%";
  $("#dist-moderate").style.width = (d.moderate / tot * 100) + "%";
  $("#dist-minor").style.width = (d.minor / tot * 100) + "%";
  $("#lg-severe").textContent = d.severe;
  $("#lg-moderate").textContent = d.moderate;
  $("#lg-minor").textContent = d.minor;

  if (data.overlay) { $("#result-img").src = data.overlay; $("#result-img").hidden = false; }
  else $("#result-img").hidden = true;

  $("#result-rows").innerHTML = data.detections.map((x) => {
    const [a, b, c, e] = x.box;
    return `<tr><td><b>${x.label}</b></td>
      <td><span class="badge ${x.severity}">${SEV_KO[x.severity]}</span></td>
      <td>${(x.confidence * 100).toFixed(0)}%</td>
      <td class="mono">${a}, ${b}, ${c - a}, ${e - b}</td></tr>`;
  }).join("");

  lastCoco = data.coco;
  $("#btn-export").disabled = false;
  showState("#result");
}

// ---------- COCO 내보내기 ----------
$("#btn-export").addEventListener("click", () => {
  if (!lastCoco) return;
  const blob = new Blob([JSON.stringify(lastCoco, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "annotations_coco.json"; a.click();
});

// ---------- 히스토리 (localStorage) ----------
function loadHistory() { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } }
function saveHistory(data) {
  const hist = loadHistory();
  hist.unshift({
    t: Date.now(), query: $("#query").value || "(질의 없음)",
    model: data.stats.model, count: data.stats.count,
    overlay: data.overlay, detections: data.detections,
    severity_dist: data.severity_dist, stats: data.stats, coco: data.coco,
  });
  localStorage.setItem(HKEY, JSON.stringify(hist.slice(0, 12))); // 용량 제한
}
function timeAgo(ts) {
  const m = Math.floor((Date.now() - ts) / 60000);
  if (m < 1) return "방금"; if (m < 60) return m + "분 전";
  const h = Math.floor(m / 60); if (h < 24) return h + "시간 전";
  return Math.floor(h / 24) + "일 전";
}
function renderHistory() {
  const hist = loadHistory(), grid = $("#history-grid");
  $("#history-empty").hidden = hist.length > 0;
  grid.innerHTML = hist.map((h, i) => `
    <div class="h-card" data-i="${i}">
      <img src="${h.overlay || ""}" alt="">
      <div class="h-meta"><div class="h-q">${h.query}</div>
        <div class="h-sub"><span>${h.model} · ${h.count}건</span><span>${timeAgo(h.t)}</span></div></div>
    </div>`).join("");
  grid.querySelectorAll(".h-card").forEach((card) => card.addEventListener("click", () => {
    const h = hist[+card.dataset.i];
    document.querySelector('.tab[data-view="detect"]').click();
    if (h.overlay) showPreview(h.overlay);
    renderResult(h);
  }));
}
$("#btn-clear-history").addEventListener("click", () => {
  if (confirm("히스토리를 모두 삭제할까요?")) { localStorage.removeItem(HKEY); renderHistory(); }
});
