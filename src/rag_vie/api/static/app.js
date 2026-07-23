const HISTORY_KEY = "ragvie_history";
const CONFIG_KEY = "ragvie_config";
const THEME_KEY = "ragvie_theme";
const MAX_HISTORY = 30;

const els = {
  healthDot: document.getElementById("healthDot"),
  themeToggle: document.getElementById("themeToggle"),
  indexDir: document.getElementById("indexDir"),
  mlpPath: document.getElementById("mlpPath"),
  useGenerator: document.getElementById("useGenerator"),
  configHint: document.getElementById("configHint"),
  refreshConfig: document.getElementById("refreshConfig"),
  sampleQueries: document.getElementById("sampleQueries"),
  queryInput: document.getElementById("queryInput"),
  queryPanel: document.getElementById("queryPanel"),
  submitQuery: document.getElementById("submitQuery"),
  submitCompare: document.getElementById("submitCompare"),
  deaccentQuery: document.getElementById("deaccentQuery"),
  latency: document.getElementById("latency"),
  errorBox: document.getElementById("errorBox"),
  resultsPanel: document.getElementById("resultsPanel"),
  featuresBars: document.getElementById("featuresBars"),
  weightsBars: document.getElementById("weightsBars"),
  weightsStacked: document.getElementById("weightsStacked"),
  answerText: document.getElementById("answerText"),
  copyAnswer: document.getElementById("copyAnswer"),
  passagesList: document.getElementById("passagesList"),
  comparePanel: document.getElementById("comparePanel"),
  compareFeatures: document.getElementById("compareFeatures"),
  compareTable: document.getElementById("compareTable"),
  historyList: document.getElementById("historyList"),
  historyEmpty: document.getElementById("historyEmpty"),
  clearHistory: document.getElementById("clearHistory"),
};

const WEIGHT_ORDER = ["dense", "bm25", "sparse", "toneless"];
const WEIGHT_COLORS = { dense: "#818cf8", bm25: "#22d3ee", sparse: "#fbbf24", toneless: "#c084fc" };

// The 8 Vietnamese linguistic features the MLP router reads (FEATURE_NAMES in
// features/vietnamese.py). Labels/tooltips live here so the API stays label-free.
const FEATURE_META = {
  diacritic_ratio:   { label: "Tỉ lệ có dấu",  desc: "Tỉ lệ âm tiết có dấu thanh. Thấp → router dồn trọng số sang kênh toneless." },
  compound_ratio:    { label: "Từ ghép",       desc: "Tỉ lệ từ ghép sau word segmentation. Cao → có lợi cho BM25." },
  english_ratio:     { label: "Tiếng Anh",     desc: "Tỉ lệ token tiếng Anh (code-switching). Cao → có lợi cho sparse (BGE-M3)." },
  tech_term_ratio:   { label: "Thuật ngữ KT",  desc: "Tỉ lệ từ kỹ thuật / chuyên ngành trong câu hỏi." },
  clause_count_norm: { label: "Số mệnh đề",    desc: "Số mệnh đề (proxy cho câu hỏi multi-hop), chuẩn hoá ở 5." },
  has_question_word: { label: "Từ để hỏi",     desc: "Có từ để hỏi (ai, gì, nào, tại sao…) hay không (0/1)." },
  query_length_norm: { label: "Độ dài",        desc: "Độ dài câu hỏi, chuẩn hoá ở 20 âm tiết." },
  oov_ratio:         { label: "OOV",           desc: "Tỉ lệ token không có trong từ điển BM25 của corpus." },
};
const FEATURE_ORDER = Object.keys(FEATURE_META);

// Strip Vietnamese diacritics for the "Bỏ dấu" demo. NFD splits base letters
// from combining tone marks; đ/Đ are not decomposed by NFD so map them explicitly.
function removeDiacritics(s) {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "")
          .replace(/đ/g, "d").replace(/Đ/g, "D");
}

// ── Theme ────────────────────────────────────────────────────────────────────
function getTheme() {
  return localStorage.getItem(THEME_KEY) || "dark";
}
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  els.themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
  localStorage.setItem(THEME_KEY, theme);
}
els.themeToggle.addEventListener("click", () => {
  applyTheme(getTheme() === "dark" ? "light" : "dark");
});
applyTheme(getTheme());

// ── Config persistence ───────────────────────────────────────────────────────
function loadSavedConfig() {
  try {
    const saved = JSON.parse(localStorage.getItem(CONFIG_KEY) || "{}");
    if (saved.indexDir) els.indexDir.value = saved.indexDir;
    if (saved.mlpPath) els.mlpPath.value = saved.mlpPath;
    if (typeof saved.useGenerator === "boolean") els.useGenerator.checked = saved.useGenerator;
  } catch { /* ignore */ }
}

function saveConfig() {
  localStorage.setItem(CONFIG_KEY, JSON.stringify({
    indexDir: els.indexDir.value,
    mlpPath: els.mlpPath.value,
    useGenerator: els.useGenerator.checked,
  }));
}

// ── History ──────────────────────────────────────────────────────────────────
function readHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"); }
  catch { return []; }
}
function writeHistory(entries) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
}
function pushHistory(entry) {
  const entries = readHistory();
  entries.unshift(entry);
  writeHistory(entries);
  renderHistory();
}

function relativeTime(ts) {
  const diff = Math.floor((Date.now() - ts) / 1000);
  if (diff < 60) return "vừa xong";
  if (diff < 3600) return `${Math.floor(diff / 60)} phút trước`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} giờ trước`;
  return new Date(ts).toLocaleDateString("vi-VN");
}

function renderHistory() {
  const entries = readHistory();
  els.historyList.innerHTML = "";
  els.historyEmpty.hidden = entries.length > 0;
  for (const entry of entries) {
    const li = document.createElement("li");
    li.className = "history-item";
    li.dataset.id = entry.id;
    li.innerHTML = `
      <div class="history-query">${escapeHtml(entry.query)}</div>
      <div class="history-meta">
        <span class="history-mode-tag">${entry.mode}</span>
        <span>${relativeTime(entry.ts)}</span>
      </div>`;
    li.addEventListener("click", () => restoreHistoryEntry(entry));
    els.historyList.appendChild(li);
  }
}

function restoreHistoryEntry(entry) {
  els.queryInput.value = entry.query;
  if (entry.index_dir) els.indexDir.value = entry.index_dir;
  if (entry.mlp_path) els.mlpPath.value = entry.mlp_path;
  els.useGenerator.checked = entry.use_generator;
  clearError();
  if (entry.mode === "compare") {
    renderCompare(entry.result);
    els.comparePanel.hidden = false;
    els.resultsPanel.hidden = true;
  } else {
    renderSingle(entry.result);
    els.resultsPanel.hidden = false;
    els.comparePanel.hidden = true;
  }
  els.latency.textContent = `${entry.result.latency_ms} ms (lịch sử)`;
  document.querySelectorAll(".history-item")
    .forEach((el) => el.classList.toggle("active", el.dataset.id === entry.id));
}

els.clearHistory.addEventListener("click", () => { writeHistory([]); renderHistory(); });

// ── Health / Config ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error();
    els.healthDot.classList.add("ok");
    els.healthDot.classList.remove("down");
    els.healthDot.title = "API OK";
  } catch {
    els.healthDot.classList.add("down");
    els.healthDot.classList.remove("ok");
    els.healthDot.title = "API không phản hồi";
  }
}

function fillSelect(select, values, defaultVal) {
  const currentVal = select.value;
  select.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  }
  if (currentVal && values.includes(currentVal)) {
    select.value = currentVal;
  } else if (defaultVal && values.includes(defaultVal)) {
    select.value = defaultVal;
  }
}

async function loadConfig() {
  try {
    const res = await fetch("/api/indexes");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const indexPaths = data.index_dirs.map((n) => `${data.default_index_dir}/${n}`);
    const mlpPaths = data.mlp_checkpoints.map((n) => `checkpoints/${n}`);
    fillSelect(els.indexDir, indexPaths, data.default_index_dir);
    fillSelect(els.mlpPath, mlpPaths, data.default_mlp_path);

    // Restore saved config after populating selects
    loadSavedConfig();

    if (data.index_dirs.length === 0) {
      els.configHint.textContent = "Không tìm thấy index — trỏ INDEX_DIR trong .env hoặc copy thư mục index về đây.";
    } else {
      els.configHint.textContent = `${data.index_dirs.length} index · ${data.mlp_checkpoints.length} checkpoint`;
    }
  } catch (err) {
    els.configHint.textContent = `Lỗi: ${err.message}`;
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.hidden = false;
}
function clearError() {
  els.errorBox.hidden = true;
  els.errorBox.textContent = "";
}

// ── Feature Bars (router explainability) ─────────────────────────────────────
function renderFeatures(container, features) {
  container.innerHTML = "";
  if (!features || Object.keys(features).length === 0) {
    container.innerHTML =
      '<p class="hint">Checkpoint này không trả về đặc trưng ngôn ngữ.</p>';
    return;
  }
  const present = FEATURE_ORDER.filter((k) => k in features);
  present.forEach((key, i) => {
    const meta = FEATURE_META[key];
    const value = features[key];
    const row = document.createElement("div");
    row.className = "feature-row" + (key === "diacritic_ratio" ? " feature-key" : "");
    row.title = meta.desc;
    row.innerHTML = `
      <span class="feature-label">${meta.label}</span>
      <div class="feature-bar-track">
        <div class="feature-bar-fill" style="width: 0%"></div>
      </div>
      <span class="feature-val">${value.toFixed(2)}</span>`;
    container.appendChild(row);
    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".feature-bar-fill").style.width =
          `${Math.max(0, Math.min(1, value)) * 100}%`;
      }, i * 60);
    });
  });
}

// ── Weight Bars ──────────────────────────────────────────────────────────────
function renderWeightBars(container, weights) {
  container.innerHTML = "";
  const present = WEIGHT_ORDER.filter((k) => k in weights);
  for (let i = 0; i < present.length; i++) {
    const key = present[i];
    const value = weights[key];
    const row = document.createElement("div");
    row.className = "weight-row";
    row.innerHTML = `
      <span class="weight-label">${key}</span>
      <div class="weight-bar-track">
        <div class="weight-bar-fill ${key}" style="width: 0%"></div>
      </div>
      <span class="weight-pct">${(value * 100).toFixed(1)}%</span>`;
    container.appendChild(row);
    // Animate fill
    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".weight-bar-fill").style.width =
          `${Math.max(0, Math.min(1, value)) * 100}%`;
      }, i * 100);
    });
  }
}

function renderStackedBar(container, weights) {
  container.innerHTML = "";
  const present = WEIGHT_ORDER.filter((k) => k in weights);
  const total = present.reduce((s, k) => s + weights[k], 0) || 1;
  for (const key of present) {
    const seg = document.createElement("div");
    seg.className = `stacked-seg ${key}`;
    seg.style.width = "0%";
    seg.title = `${key}: ${(weights[key] * 100).toFixed(1)}%`;
    container.appendChild(seg);
    requestAnimationFrame(() => {
      setTimeout(() => { seg.style.width = `${(weights[key] / total) * 100}%`; }, 150);
    });
  }
}

// ── Passages ─────────────────────────────────────────────────────────────────
function renderPassages(passages) {
  els.passagesList.innerHTML = "";
  passages.forEach((p, idx) => {
    const li = document.createElement("li");
    li.className = "passage-item";
    li.innerHTML = `
      <div class="passage-meta">
        <span><span class="passage-rank">${idx + 1}</span>
          <span class="passage-id">${escapeHtml(p.id)}</span></span>
        <span class="passage-score">${p.score.toFixed(4)}</span>
      </div>
      <p class="passage-text">${escapeHtml(p.passage)}</p>`;
    li.addEventListener("click", () => li.classList.toggle("expanded"));
    els.passagesList.appendChild(li);
  });
}

function renderSingle(data) {
  renderFeatures(els.featuresBars, data.features);
  renderWeightBars(els.weightsBars, data.weights);
  renderStackedBar(els.weightsStacked, data.weights);
  els.answerText.textContent = data.answer || "(generator tắt hoặc không có câu trả lời)";
  renderPassages(data.retrieved);
}

// ── Copy ─────────────────────────────────────────────────────────────────────
els.copyAnswer.addEventListener("click", async () => {
  const text = els.answerText.textContent;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    els.copyAnswer.textContent = "✓";
    els.copyAnswer.classList.add("copied");
    setTimeout(() => { els.copyAnswer.textContent = "📋"; els.copyAnswer.classList.remove("copied"); }, 1500);
  } catch { /* clipboard may fail in non-HTTPS */ }
});

// ── Compare ──────────────────────────────────────────────────────────────────
function renderCompare(data) {
  renderFeatures(els.compareFeatures, data.features);
  els.compareTable.innerHTML = "";
  const mlpMethod = data.methods.find((m) => m.name === "mlp");
  const mlpTopIds = new Set((mlpMethod?.retrieved || []).slice(0, 5).map((p) => p.id));

  for (const method of data.methods) {
    const block = document.createElement("div");
    block.className = `compare-method${method.name === "mlp" ? " is-mlp-card" : ""}`;

    const head = document.createElement("div");
    head.className = "compare-method-head";
    const name = document.createElement("span");
    name.className = `compare-method-name${method.name === "mlp" ? " is-mlp" : ""}`;
    name.textContent = method.label;
    const wInline = document.createElement("span");
    wInline.className = "compare-weights-inline";
    wInline.textContent = WEIGHT_ORDER.filter((k) => k in method.weights)
      .map((k) => `${k}=${method.weights[k].toFixed(2)}`).join("  ");
    head.append(name, wInline);

    const list = document.createElement("ol");
    list.className = "compare-passages";
    for (const p of method.retrieved.slice(0, 5)) {
      const li = document.createElement("li");
      li.className = "compare-passage";
      const dot = document.createElement("span");
      if (method.name !== "mlp" && mlpTopIds.has(p.id)) {
        dot.className = "match-dot";
        dot.title = "Trùng top-5 MLP";
      }
      const score = document.createElement("span");
      score.className = "compare-passage-score";
      score.textContent = p.score.toFixed(3);
      const text = document.createElement("span");
      text.className = "compare-passage-text";
      text.title = p.passage;
      text.textContent = p.passage;
      li.append(dot, score, text);
      list.appendChild(li);
    }
    block.append(head, list);

    if (method.answer) {
      const answer = document.createElement("div");
      answer.className = "compare-answer";
      answer.textContent = method.answer;
      block.appendChild(answer);
    }
    els.compareTable.appendChild(block);
  }
}

// ── Loading overlay ──────────────────────────────────────────────────────────
function showLoading() {
  let overlay = els.queryPanel.querySelector(".loading-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "loading-overlay";
    overlay.innerHTML = '<div class="spinner"></div>';
    els.queryPanel.style.position = "relative";
    els.queryPanel.appendChild(overlay);
  }
}
function hideLoading() {
  const overlay = els.queryPanel.querySelector(".loading-overlay");
  if (overlay) overlay.remove();
}

// ── Query execution ──────────────────────────────────────────────────────────
function currentRequestBody() {
  return {
    query: els.queryInput.value.trim(),
    index_dir: els.indexDir.value || null,
    mlp_path: els.mlpPath.value || null,
    use_generator: els.useGenerator.checked,
  };
}

async function runRequest(mode) {
  const body = currentRequestBody();
  if (!body.query) { showError("Nhập câu hỏi trước đã."); return; }

  clearError(); saveConfig();
  els.resultsPanel.hidden = true;
  els.comparePanel.hidden = true;
  els.latency.textContent = "";
  els.submitQuery.disabled = true;
  els.submitCompare.disabled = true;
  els.deaccentQuery.disabled = true;
  showLoading();

  try {
    const res = await fetch(mode === "compare" ? "/api/compare" : "/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);

    if (mode === "compare") {
      renderCompare(data);
      els.comparePanel.hidden = false;
    } else {
      renderSingle(data);
      els.resultsPanel.hidden = false;
    }
    els.latency.textContent = `${data.latency_ms} ms`;

    pushHistory({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      ts: Date.now(), mode,
      query: body.query,
      index_dir: data.index_dir,
      mlp_path: data.mlp_path,
      use_generator: body.use_generator,
      result: data,
    });
  } catch (err) {
    showError(err.message);
  } finally {
    els.submitQuery.disabled = false;
    els.submitCompare.disabled = false;
    els.deaccentQuery.disabled = false;
    hideLoading();
  }
}

// ── Sample queries ───────────────────────────────────────────────────────────
els.sampleQueries.addEventListener("click", (e) => {
  const btn = e.target.closest(".sample-btn");
  if (!btn) return;
  els.queryInput.value = btn.dataset.query;
  els.queryInput.focus();
});

// ── Event wiring ─────────────────────────────────────────────────────────────
els.submitQuery.addEventListener("click", () => runRequest("query"));
els.submitCompare.addEventListener("click", () => runRequest("compare"));
els.deaccentQuery.addEventListener("click", () => {
  const q = els.queryInput.value.trim();
  if (!q) { showError("Nhập câu hỏi trước đã."); return; }
  els.queryInput.value = removeDiacritics(q);
  runRequest("query");
});
els.queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    runRequest(e.shiftKey ? "compare" : "query");
  }
});
els.refreshConfig.addEventListener("click", loadConfig);
[els.indexDir, els.mlpPath, els.useGenerator].forEach((el) =>
  el.addEventListener("change", saveConfig)
);

// ── Init ─────────────────────────────────────────────────────────────────────
renderHistory();
checkHealth();
loadConfig();
setInterval(checkHealth, 30000);
