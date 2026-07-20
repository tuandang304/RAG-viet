const HISTORY_KEY = "ragvie_history";
const CONFIG_KEY = "ragvie_config";
const MAX_HISTORY = 30;

const els = {
  healthDot: document.getElementById("healthDot"),
  indexDir: document.getElementById("indexDir"),
  indexDirList: document.getElementById("indexDirList"),
  mlpPath: document.getElementById("mlpPath"),
  mlpPathList: document.getElementById("mlpPathList"),
  useGenerator: document.getElementById("useGenerator"),
  configHint: document.getElementById("configHint"),
  refreshConfig: document.getElementById("refreshConfig"),
  queryInput: document.getElementById("queryInput"),
  submitQuery: document.getElementById("submitQuery"),
  submitCompare: document.getElementById("submitCompare"),
  latency: document.getElementById("latency"),
  errorBox: document.getElementById("errorBox"),
  resultsPanel: document.getElementById("resultsPanel"),
  weightsBars: document.getElementById("weightsBars"),
  answerText: document.getElementById("answerText"),
  passagesList: document.getElementById("passagesList"),
  comparePanel: document.getElementById("comparePanel"),
  compareTable: document.getElementById("compareTable"),
  historyList: document.getElementById("historyList"),
  historyEmpty: document.getElementById("historyEmpty"),
  clearHistory: document.getElementById("clearHistory"),
};

const WEIGHT_ORDER = ["dense", "bm25", "sparse", "toneless"];

// ── Config persistence ──────────────────────────────────────────────────────

function loadSavedConfig() {
  try {
    const saved = JSON.parse(localStorage.getItem(CONFIG_KEY) || "{}");
    if (saved.indexDir) els.indexDir.value = saved.indexDir;
    if (saved.mlpPath) els.mlpPath.value = saved.mlpPath;
    if (typeof saved.useGenerator === "boolean") els.useGenerator.checked = saved.useGenerator;
  } catch {
    // ignore malformed localStorage content
  }
}

function saveConfig() {
  localStorage.setItem(
    CONFIG_KEY,
    JSON.stringify({
      indexDir: els.indexDir.value,
      mlpPath: els.mlpPath.value,
      useGenerator: els.useGenerator.checked,
    })
  );
}

// ── History ──────────────────────────────────────────────────────────────────

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
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

function renderHistory() {
  const entries = readHistory();
  els.historyList.innerHTML = "";
  els.historyEmpty.hidden = entries.length > 0;

  for (const entry of entries) {
    const li = document.createElement("li");
    li.className = "history-item";
    li.dataset.id = entry.id;

    const q = document.createElement("div");
    q.className = "history-query";
    q.textContent = entry.query;

    const meta = document.createElement("div");
    meta.className = "history-meta";
    const time = new Date(entry.ts).toLocaleTimeString();
    meta.innerHTML = `<span class="history-mode-tag">${entry.mode}</span><span>${time}</span>`;

    li.append(q, meta);
    li.addEventListener("click", () => restoreHistoryEntry(entry));
    els.historyList.appendChild(li);
  }
}

function restoreHistoryEntry(entry) {
  els.queryInput.value = entry.query;
  els.indexDir.value = entry.index_dir;
  els.mlpPath.value = entry.mlp_path;
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
  els.latency.textContent = `${entry.result.latency_ms} ms (từ lịch sử)`;

  document
    .querySelectorAll(".history-item")
    .forEach((el) => el.classList.toggle("active", el.dataset.id === entry.id));
}

els.clearHistory.addEventListener("click", () => {
  writeHistory([]);
  renderHistory();
});

// ── Health / config ──────────────────────────────────────────────────────────

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

function fillDatalist(datalist, values) {
  datalist.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    datalist.appendChild(opt);
  }
}

async function loadConfig() {
  try {
    const res = await fetch("/api/indexes");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (!els.indexDir.value) els.indexDir.value = data.default_index_dir;
    if (!els.mlpPath.value) els.mlpPath.value = data.default_mlp_path;

    const indexDirPaths = data.index_dirs.map((name) => `${data.default_index_dir}/${name}`);
    fillDatalist(els.indexDirList, indexDirPaths);
    fillDatalist(
      els.mlpPathList,
      data.mlp_checkpoints.map((name) => `checkpoints/${name}`)
    );

    if (data.index_dirs.length === 0) {
      els.configHint.textContent =
        "Không tìm thấy index nào trong thư mục indexes/ — data có thể đang ở máy khác. " +
        "Copy thư mục index về đây hoặc trỏ INDEX_DIR trong .env, rồi bấm Làm mới.";
    } else {
      els.configHint.textContent = `${data.index_dirs.length} index, ${data.mlp_checkpoints.length} checkpoint tìm thấy.`;
    }
  } catch (err) {
    els.configHint.textContent = `Không tải được cấu hình: ${err.message}`;
  }
}

// ── Single-query rendering ──────────────────────────────────────────────────

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.hidden = false;
}

function clearError() {
  els.errorBox.hidden = true;
  els.errorBox.textContent = "";
}

function renderWeightBars(container, weights) {
  container.innerHTML = "";
  const present = WEIGHT_ORDER.filter((k) => k in weights);
  for (const key of present) {
    const value = weights[key];
    const row = document.createElement("div");
    row.className = "weight-row";

    const label = document.createElement("span");
    label.textContent = key;

    const track = document.createElement("div");
    track.className = "weight-bar-track";
    const fill = document.createElement("div");
    fill.className = `weight-bar-fill ${key}`;
    fill.style.width = `${Math.max(0, Math.min(1, value)) * 100}%`;
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.textContent = `${(value * 100).toFixed(1)}%`;

    row.append(label, track, pct);
    container.appendChild(row);
  }
}

function renderPassages(passages) {
  els.passagesList.innerHTML = "";
  for (const p of passages) {
    const li = document.createElement("li");
    li.className = "passage-item";

    const meta = document.createElement("div");
    meta.className = "passage-meta";
    meta.innerHTML = `<span>id: ${p.id}</span><span>score: ${p.score.toFixed(4)}</span>`;

    const text = document.createElement("p");
    text.className = "passage-text";
    text.textContent = p.passage;

    li.append(meta, text);
    els.passagesList.appendChild(li);
  }
}

function renderSingle(data) {
  renderWeightBars(els.weightsBars, data.weights);
  els.answerText.textContent = data.answer || "(generator tắt hoặc không có câu trả lời)";
  renderPassages(data.retrieved);
}

// ── Compare rendering ────────────────────────────────────────────────────────

function renderCompare(data) {
  els.compareTable.innerHTML = "";
  const mlpMethod = data.methods.find((m) => m.name === "mlp");
  const mlpTopIds = new Set((mlpMethod?.retrieved || []).slice(0, 5).map((p) => p.id));

  for (const method of data.methods) {
    const block = document.createElement("div");
    block.className = "compare-method";

    const head = document.createElement("div");
    head.className = "compare-method-head";
    const name = document.createElement("span");
    name.className = `compare-method-name${method.name === "mlp" ? " is-mlp" : ""}`;
    name.textContent = method.label;
    const weightsInline = document.createElement("span");
    weightsInline.className = "compare-weights-inline";
    weightsInline.textContent = WEIGHT_ORDER.filter((k) => k in method.weights)
      .map((k) => `${k}=${method.weights[k].toFixed(2)}`)
      .join("  ");
    head.append(name, weightsInline);

    const list = document.createElement("ol");
    list.className = "compare-passages";
    for (const p of method.retrieved.slice(0, 5)) {
      const li = document.createElement("li");
      li.className = "compare-passage";
      const dot = document.createElement("span");
      if (method.name !== "mlp" && mlpTopIds.has(p.id)) {
        dot.className = "match-dot";
        dot.title = "Cũng nằm trong top-5 của MLP";
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

// ── Query execution ──────────────────────────────────────────────────────────

function currentRequestBody() {
  const query = els.queryInput.value.trim();
  return {
    query,
    index_dir: els.indexDir.value || null,
    mlp_path: els.mlpPath.value || null,
    use_generator: els.useGenerator.checked,
  };
}

async function runRequest(mode) {
  const body = currentRequestBody();
  if (!body.query) {
    showError("Nhập câu hỏi trước đã.");
    return;
  }

  clearError();
  saveConfig();
  els.resultsPanel.hidden = true;
  els.comparePanel.hidden = true;
  els.latency.textContent = "";
  const button = mode === "compare" ? els.submitCompare : els.submitQuery;
  const otherButton = mode === "compare" ? els.submitQuery : els.submitCompare;
  const originalLabel = button.textContent;
  button.disabled = true;
  otherButton.disabled = true;
  button.textContent = "Đang chạy…";

  try {
    const res = await fetch(mode === "compare" ? "/api/compare" : "/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }

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
      ts: Date.now(),
      mode,
      query: body.query,
      index_dir: data.index_dir,
      mlp_path: data.mlp_path,
      use_generator: body.use_generator,
      result: data,
    });
  } catch (err) {
    showError(err.message);
  } finally {
    button.disabled = false;
    otherButton.disabled = false;
    button.textContent = originalLabel;
  }
}

els.submitQuery.addEventListener("click", () => runRequest("query"));
els.submitCompare.addEventListener("click", () => runRequest("compare"));
els.queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runRequest("query");
});
els.refreshConfig.addEventListener("click", loadConfig);
[els.indexDir, els.mlpPath, els.useGenerator].forEach((el) =>
  el.addEventListener("change", saveConfig)
);

loadSavedConfig();
renderHistory();
checkHealth();
loadConfig();
