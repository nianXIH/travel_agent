const API = "/api/v1";

const STATUS_LABELS = {
  imported: "已导入",
  failed: "失败",
  deleted: "已删除",
  queued: "排队中",
  running: "执行中",
  success: "成功",
  partial: "部分失败",
  cancelled: "已取消",
  stale: "已中断",
};

const PAGE_TITLES = {
  overview: ["概览", "知识库运行状态与内容分布"],
  documents: ["知识文档", "按状态、城市与内容类型管理导入结果"],
  tasks: ["导入任务", "查看进度、失败文件与重试"],
  ingest: ["上传与扫描", "新增资料并提交导入任务"],
  search: ["检索测试", "验证知识库召回与引用"],
};

const state = {
  tab: "overview",
  token: sessionStorage.getItem("adminToken") || "",
  tenantId: sessionStorage.getItem("adminTenant") || "",
  isGlobalAdmin: false,
  docs: { limit: 20, offset: 0, total: 0 },
  tasks: { limit: 20, offset: 0, total: 0, status: "" },
  detailDocId: "",
  toastTimer: null,
};

const $ = (id) => document.getElementById(id);
const qsa = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

function formatTime(value) {
  if (!value) return "—";
  if (typeof value === "number") {
    return new Date(value * 1000).toLocaleString("zh-CN", { hour12: false });
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("zh-CN", { hour12: false });
}

function formatBytes(size) {
  const value = Number(size) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(2)} MB`;
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 10 ? `${text.slice(0, 8)}…` : text || "—";
}

function statusBadge(status) {
  const key = String(status || "unknown");
  const label = STATUS_LABELS[key] || key;
  return `<span class="badge ${escapeHtml(key)}">${escapeHtml(label)}</span>`;
}

function toast(message, isError = false) {
  const element = $("toast");
  element.textContent = message;
  element.classList.toggle("error", isError);
  element.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => element.classList.remove("show"), 3200);
}

async function api(path, options = {}) {
  const { skipTenant = false, ...fetchOptions } = options;
  const headers = { ...(fetchOptions.headers || {}) };
  if (state.token) {
    headers["X-Admin-Token"] = state.token;
    headers["X-API-Key"] = state.token;
  }
  if (!skipTenant && state.tenantId) headers["X-Tenant-Id"] = state.tenantId;
  const isForm = fetchOptions.body instanceof FormData;
  if (fetchOptions.body && !isForm) headers["Content-Type"] = "application/json";

  const response = await fetch(API + path, { ...fetchOptions, headers });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!response.ok) {
    if (response.status === 401) {
      toast("需要正确的管理令牌", true);
      $("admin-token").focus();
    }
    const detail = data && (data.message || data.detail);
    const message = detail
      ? typeof detail === "string" ? detail : JSON.stringify(detail)
      : `请求失败（${response.status}）`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return data;
}

/* Navigation */
function switchTab(tab, updateHash = true) {
  state.tab = tab;
  if (updateHash && window.location.hash !== `#${tab}`) {
    window.history.replaceState(null, "", `#${tab}`);
  }
  qsa(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.tab === tab));
  qsa(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${tab}`));
  const [title, subtitle] = PAGE_TITLES[tab] || ["管理后台", ""];
  $("page-title").textContent = title;
  $("page-subtitle").textContent = subtitle;
  loadTab(tab);
}

function loadTab(tab) {
  const loaders = {
    overview: loadOverview,
    documents: loadDocuments,
    tasks: loadTasks,
    ingest: renderIngestResult,
    search: () => {},
  };
  (loaders[tab] || (() => {}))();
}

/* Readiness */
async function loadReadiness() {
  const pill = $("readiness");
  try {
    const response = await fetch("/readyz");
    const data = await response.json();
    if (data.status === "ready") {
      pill.textContent = `就绪 · ${data.chunk_count ?? 0} 块`;
      pill.className = "pill ok";
    } else {
      pill.textContent = "降级";
      pill.className = "pill bad";
      pill.title = data.milvus_error || data.mongo_error || "";
    }
  } catch {
    pill.textContent = "无法连接";
    pill.className = "pill bad";
  }
}

/* Tenants */
async function loadTenants() {
  const picker = $("tenant-picker");
  const select = $("tenant-select");
  try {
    const data = await api("/admin/tenants", { skipTenant: true });
    state.isGlobalAdmin = data.scope === "all";
    if (state.isGlobalAdmin) {
      const options = ['<option value="">全部租户</option>'].concat(
        data.tenants.map(
          (tenant) =>
            `<option value="${escapeHtml(tenant.tenant_id)}">${escapeHtml(tenant.tenant_name)}</option>`
        )
      );
      select.innerHTML = options.join("");
      if (!data.tenants.some((tenant) => tenant.tenant_id === state.tenantId)) {
        state.tenantId = "";
      }
      select.value = state.tenantId;
      picker.hidden = false;
    } else {
      state.tenantId = data.tenant_id || "";
      select.innerHTML = "";
      picker.hidden = true;
    }
  } catch (error) {
    state.isGlobalAdmin = false;
    state.tenantId = "";
    sessionStorage.removeItem("adminTenant");
    select.innerHTML = "";
    picker.hidden = true;
    if (!error || error.status !== 401) toast("无法读取租户列表", true);
  }
  updateTenantControls();
}

function tenantParamString() {
  return state.tenantId ? `tenant_id=${encodeURIComponent(state.tenantId)}` : "";
}

function updateTenantControls() {
  const blocked = !state.tenantId;
  [
    "upload-file",
    "upload-btn",
    "scan-force",
    "scan-btn",
    "search-query",
    "search-mode",
    "search-topk",
    "search-rerank",
    "search-city",
    "search-type",
    "search-btn",
  ].forEach((id) => {
    const element = $(id);
    if (element) element.disabled = blocked;
  });
  $("tenant-guard").hidden = !blocked;
  $("search-tenant-guard").hidden = !blocked;
}

/* Overview */
async function loadOverview() {
  const cards = $("overview-cards");
  cards.innerHTML = '<div class="card muted">加载中…</div>';
  try {
    const suffix = tenantParamString();
    const data = await api(`/admin/overview${suffix ? `?${suffix}` : ""}`);
    const status = data.documents_by_status || {};
    const tasks = data.tasks_by_status || {};
    const chunkValue = data.chunks === null || data.chunks === undefined ? "—" : data.chunks;
    const chunkNote = data.chunk_error ? `Milvus 不可用：${escapeHtml(data.chunk_error)}` : "向量库分块总数";

    cards.innerHTML = [
      statCard("文档总数", data.total_documents, `已导入 ${status.imported || 0} · 删除 ${status.deleted || 0}`),
      statCard("已导入", status.imported || 0, "可用于检索与问答"),
      statCard("失败文档", status.failed || 0, "可在任务页查看失败原因"),
      statCard("向量分块", chunkValue, chunkNote),
      statCard("导入任务", data.total_tasks, `执行中 ${(tasks.running || 0) + (tasks.queued || 0)}`),
    ].join("");

    renderBars("overview-status", status, data.total_documents);
    renderBars("task-status", tasks, data.total_tasks);
    renderChips("cities", data.cities);
    renderChips("types", data.content_types);
  } catch (error) {
    cards.innerHTML = `<div class="card muted">${escapeHtml(error.message)}</div>`;
  }
}

function statCard(label, value, note) {
  return `<div class="card">
    <div class="stat-label">${escapeHtml(label)}</div>
    <div class="stat-value">${escapeHtml(value)}</div>
    <div class="stat-note">${note || ""}</div>
  </div>`;
}

function renderBars(targetId, counts, total) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  const container = $(targetId);
  if (!entries.length) {
    container.innerHTML = '<p class="muted">暂无数据</p>';
    return;
  }
  const sum = total || entries.reduce((acc, [, count]) => acc + count, 0) || 1;
  container.innerHTML = entries.map(([key, count]) => {
    const percent = Math.round((count / sum) * 100);
    return `<div class="bar-row">
      <span>${escapeHtml(STATUS_LABELS[key] || key)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${percent}%"></div></div>
      <span class="num">${count}</span>
    </div>`;
  }).join("");
}

function renderChips(targetId, counts) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  const container = $(targetId);
  container.innerHTML = entries.length
    ? entries.map(([key, count]) => `<span class="chip">${escapeHtml(key)} · ${count}</span>`).join("")
    : '<span class="muted">暂无数据</span>';
}

/* Documents */
function readDocFilters() {
  return {
    keyword: $("doc-keyword").value.trim(),
    status: $("doc-status").value,
    city: $("doc-city").value.trim(),
    content_type: $("doc-type").value.trim(),
  };
}

function documentParams() {
  const filters = readDocFilters();
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value) params.set(key, value);
  });
  params.set("limit", state.docs.limit);
  params.set("offset", state.docs.offset);
  if (state.tenantId) params.set("tenant_id", state.tenantId);
  return params;
}

async function loadDocuments() {
  const rows = $("doc-rows");
  rows.innerHTML = '<tr><td colspan="8" class="empty">加载中…</td></tr>';
  try {
    const data = await api(`/admin/documents?${documentParams().toString()}`);
    state.docs.total = data.total;
    $("doc-count").textContent = `共 ${data.total} 份文档`;
    const from = data.total ? data.offset + 1 : 0;
    const to = Math.min(data.offset + data.limit, data.total);
    $("doc-page").textContent = `${from}–${to}`;
    $("doc-prev").disabled = data.offset <= 0;
    $("doc-next").disabled = data.offset + data.limit >= data.total;

    if (!data.items.length) {
      rows.innerHTML = '<tr><td colspan="8" class="empty">没有匹配的文档</td></tr>';
      return;
    }
    rows.innerHTML = data.items.map((item) => `<tr>
      <td>
        <div class="cell-title">${escapeHtml(item.title || item.source_file || "未命名")}</div>
        <div class="cell-sub">${escapeHtml(item.source_path)}</div>
      </td>
      <td>${escapeHtml(item.tenant_id || "—")}</td>
      <td>${escapeHtml(item.city || "—")}</td>
      <td>${escapeHtml(item.content_type || "—")}</td>
      <td class="num">${item.chunk_count}</td>
      <td>${statusBadge(item.status)}</td>
      <td>${escapeHtml(formatTime(item.updated_at || item.imported_at))}</td>
      <td><button class="ghost small" type="button" data-doc-id="${escapeHtml(item.doc_id)}">查看</button></td>
    </tr>`).join("");
  } catch (error) {
    rows.innerHTML = `<tr><td colspan="8" class="empty">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function openDocument(docId) {
  const panel = $("doc-detail");
  state.detailDocId = docId;
  panel.classList.remove("hidden");
  panel.innerHTML = '<p class="muted">加载文档详情…</p>';
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const data = await api(`/admin/documents/${encodeURIComponent(docId)}`);
    const chunks = data.chunks || [];
    const chunkHtml = chunks.length
      ? `<div class="chunk-list">${chunks.map((chunk) => `<div class="chunk">
          <div class="chunk-head">
            <span>#${chunk.chunk_index} ${escapeHtml(chunk.section_path || "")}</span>
            <span>${escapeHtml(chunk.chunk_id)}</span>
          </div>
          <div class="chunk-body">${escapeHtml(chunk.content)}</div>
        </div>`).join("")}</div>`
      : `<p class="muted">${data.chunk_error ? `分块读取失败：${escapeHtml(data.chunk_error)}` : "没有可展示的分块。"}</p>`;

    panel.innerHTML = `
      <div class="detail-head">
        <div>
          <h2>${escapeHtml(data.title || data.source_file || "未命名文档")}</h2>
          <div class="muted">${escapeHtml(data.source_path)}</div>
        </div>
        <button id="doc-detail-close" class="ghost small" type="button">关闭</button>
      </div>
      <dl class="detail-grid">
        <div><dt>文档 ID</dt><dd>${escapeHtml(data.doc_id)}</dd></div>
        <div><dt>状态</dt><dd>${statusBadge(data.status)}</dd></div>
        <div><dt>城市</dt><dd>${escapeHtml(data.city || "—")}</dd></div>
        <div><dt>内容类型</dt><dd>${escapeHtml(data.content_type || "—")}</dd></div>
        <div><dt>分块数</dt><dd>${data.chunk_count}</dd></div>
        <div><dt>文件大小</dt><dd>${formatBytes(data.size)}</dd></div>
        <div><dt>导入时间</dt><dd>${escapeHtml(formatTime(data.imported_at))}</dd></div>
        <div><dt>更新时间</dt><dd>${escapeHtml(formatTime(data.updated_at))}</dd></div>
        <div><dt>SHA256</dt><dd>${escapeHtml(data.sha256 || "—")}</dd></div>
      </dl>
      ${data.error ? `<p class="badge failed">${escapeHtml(data.error)}</p>` : ""}
      ${chunkHtml}`;
    $("doc-detail-close").addEventListener("click", () => {
      panel.classList.add("hidden");
      state.detailDocId = "";
    });
  } catch (error) {
    panel.innerHTML = `<p class="muted">${escapeHtml(error.message)}</p>`;
  }
}

/* Tasks */
function taskParams() {
  const params = new URLSearchParams();
  if (state.tasks.status) params.set("status", state.tasks.status);
  params.set("limit", state.tasks.limit);
  params.set("offset", state.tasks.offset);
  return params;
}

async function loadTasks() {
  state.tasks.status = $("task-status-filter").value;
  const rows = $("task-rows");
  rows.innerHTML = '<tr><td colspan="12" class="empty">加载中…</td></tr>';
  try {
    const data = await api(`/ingest/tasks?${taskParams().toString()}`);
    state.tasks.total = data.total;
    $("task-count").textContent = `共 ${data.total} 个任务`;
    const offset = state.tasks.offset;
    const limit = state.tasks.limit;
    const from = data.total ? offset + 1 : 0;
    const to = Math.min(offset + limit, data.total);
    $("task-page").textContent = `${from}–${to}`;
    $("task-prev").disabled = offset <= 0;
    $("task-next").disabled = offset + limit >= data.total;

    if (!data.tasks.length) {
      rows.innerHTML = '<tr><td colspan="12" class="empty">没有任务</td></tr>';
      return;
    }
    rows.innerHTML = data.tasks.map((task) => {
      const canRetry = task.stale || (task.failed > 0 && !["queued", "running"].includes(task.status));
      return `<tr>
        <td title="${escapeHtml(task.task_id)}">${escapeHtml(shortId(task.task_id))}</td>
        <td>${escapeHtml(task.tenant_id || "—")}</td>
        <td>${escapeHtml(task.task_type)}</td>
        <td>${statusBadge(task.stale ? "stale" : task.status)}</td>
        <td class="num">${task.total}</td>
        <td class="num">${task.succeeded}</td>
        <td class="num">${task.failed}</td>
        <td class="num">${task.skipped}</td>
        <td class="cell-sub">${escapeHtml(task.current_file || "—")}</td>
        <td>${escapeHtml(formatTime(task.created_at))}</td>
        <td>${task.duration_seconds == null ? "—" : `${Number(task.duration_seconds).toFixed(1)}s`}</td>
        <td>${canRetry ? `<button class="ghost small" type="button" data-retry-id="${escapeHtml(task.task_id)}">重试</button>` : ""}</td>
      </tr>`;
    }).join("");
  } catch (error) {
    rows.innerHTML = `<tr><td colspan="12" class="empty">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function retryTask(taskId) {
  try {
    const data = await api(`/ingest/tasks/${encodeURIComponent(taskId)}/retry`, { method: "POST" });
    toast(`已提交重试任务 ${shortId(data.task_id)}`);
    loadTasks();
  } catch (error) {
    toast(error.message, true);
  }
}

/* Ingest */
function renderIngestResult() {
  const container = $("ingest-result");
  if (container.dataset.loaded === "1") return;
  container.textContent = "暂无任务";
}

function showIngestResult(label, task) {
  const container = $("ingest-result");
  container.dataset.loaded = "1";
  container.innerHTML = `<p>${escapeHtml(label)}：租户 <strong>${escapeHtml(
    state.tenantId || "default"
  )}</strong>，任务 <strong>${escapeHtml(task.task_id)}</strong> 已排队（${escapeHtml(
    task.status
  )}）。</p>
    <button class="ghost small" type="button" id="goto-task">查看任务</button>`;
  $("goto-task").addEventListener("click", () => switchTab("tasks"));
}

async function uploadDocument() {
  if (!state.tenantId) {
    toast("请先选择具体租户", true);
    return;
  }
  const input = $("upload-file");
  const file = input.files && input.files[0];
  if (!file) {
    toast("请选择要上传的文件", true);
    return;
  }
  const button = $("upload-btn");
  button.disabled = true;
  button.textContent = "上传中…";
  try {
    const form = new FormData();
    form.append("file", file);
    const task = await api("/ingest/upload", { method: "POST", body: form });
    toast(`已提交导入任务 ${shortId(task.task_id)}`);
    showIngestResult(`已上传 ${file.name}`, task);
    input.value = "";
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "上传并导入";
  }
}

async function scanDirectory() {
  if (!state.tenantId) {
    toast("请先选择具体租户", true);
    return;
  }
  const force = $("scan-force").checked;
  if (force && !window.confirm("强制扫描会重新生成所有文档的分块并覆盖同 doc_id 内容，是否继续？")) {
    return;
  }
  const button = $("scan-btn");
  button.disabled = true;
  button.textContent = "提交中…";
  try {
    const task = await api("/ingest/scan", {
      method: "POST",
      body: JSON.stringify({ force }),
    });
    toast(`已提交扫描任务 ${shortId(task.task_id)}`);
    showIngestResult(force ? "强制扫描" : "增量扫描", task);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "开始扫描";
  }
}

/* Search */
async function runSearch() {
  if (!state.tenantId) {
    toast("请先选择具体租户", true);
    return;
  }
  const query = $("search-query").value.trim();
  if (!query) {
    toast("请输入检索内容", true);
    return;
  }
  const button = $("search-btn");
  const results = $("search-results");
  button.disabled = true;
  results.innerHTML = '<div class="card muted">检索中…</div>';
  $("search-meta").textContent = "";
  try {
    const topK = Math.min(20, Math.max(1, Number($("search-topk").value) || 5));
    const body = {
      query,
      mode: $("search-mode").value,
      top_k: topK,
      rerank: $("search-rerank").checked,
    };
    const city = $("search-city").value.trim();
    const contentType = $("search-type").value.trim();
    if (city) body.city = city;
    if (contentType) body.content_type = contentType;

    const data = await api("/search", { method: "POST", body: JSON.stringify(body) });
    $("search-meta").textContent = `${data.citations.length} 条结果 · ${data.mode} · ${data.elapsed_ms} ms · ${data.reranked ? "已重排" : "未重排"}`;
    if (!data.citations.length) {
      results.innerHTML = '<div class="card muted">没有命中内容，请调整查询或知识库筛选条件。</div>';
      return;
    }
    results.innerHTML = data.citations.map((item, index) => {
      const score = item.rerank_score == null ? item.score : item.rerank_score;
      const meta = [item.city, item.content_type, item.section_path].filter(Boolean).join(" · ");
      return `<article class="result">
        <div class="result-head">
          <div>
            <div class="cell-title">#${index + 1} ${escapeHtml(item.title || item.source_file || "未命名来源")}</div>
            <div class="result-source">${escapeHtml(item.source_path || "")}${meta ? ` · ${escapeHtml(meta)}` : ""}</div>
          </div>
          <div class="result-score">score ${Number(score || 0).toFixed(4)}</div>
        </div>
        <div class="result-body">${escapeHtml(item.content)}</div>
      </article>`;
    }).join("");
  } catch (error) {
    results.innerHTML = `<div class="card muted">${escapeHtml(error.message)}</div>`;
  } finally {
    button.disabled = false;
  }
}

/* Wiring */
function bindEvents() {
  qsa(".nav-item").forEach((item) => {
    item.addEventListener("click", () => switchTab(item.dataset.tab));
  });

  $("refresh").addEventListener("click", async () => {
    loadReadiness();
    await loadTenants();
    loadTab(state.tab);
  });

  $("admin-token").value = state.token;
  $("save-token").addEventListener("click", async () => {
    state.token = $("admin-token").value.trim();
    if (state.token) sessionStorage.setItem("adminToken", state.token);
    else sessionStorage.removeItem("adminToken");
    toast("访问凭证已保存");
    await loadTenants();
    loadTab(state.tab);
  });

  $("tenant-select").addEventListener("change", () => {
    state.tenantId = $("tenant-select").value;
    if (state.tenantId) sessionStorage.setItem("adminTenant", state.tenantId);
    else sessionStorage.removeItem("adminTenant");
    state.docs.offset = 0;
    state.tasks.offset = 0;
    state.detailDocId = "";
    const detail = $("doc-detail");
    detail.classList.add("hidden");
    detail.innerHTML = "";
    updateTenantControls();
    loadReadiness();
    loadTab(state.tab);
  });

  $("doc-search").addEventListener("click", () => {
    state.docs.offset = 0;
    loadDocuments();
  });
  $("doc-reset").addEventListener("click", () => {
    $("doc-keyword").value = "";
    $("doc-status").value = "";
    $("doc-city").value = "";
    $("doc-type").value = "";
    state.docs.offset = 0;
    loadDocuments();
  });
  $("doc-keyword").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      state.docs.offset = 0;
      loadDocuments();
    }
  });
  $("doc-prev").addEventListener("click", () => {
    state.docs.offset = Math.max(0, state.docs.offset - state.docs.limit);
    loadDocuments();
  });
  $("doc-next").addEventListener("click", () => {
    state.docs.offset += state.docs.limit;
    loadDocuments();
  });
  $("doc-rows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-doc-id]");
    if (button) openDocument(button.dataset.docId);
  });

  $("task-search").addEventListener("click", () => {
    state.tasks.offset = 0;
    loadTasks();
  });
  $("task-prev").addEventListener("click", () => {
    state.tasks.offset = Math.max(0, state.tasks.offset - state.tasks.limit);
    loadTasks();
  });
  $("task-next").addEventListener("click", () => {
    state.tasks.offset += state.tasks.limit;
    loadTasks();
  });
  $("task-rows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-retry-id]");
    if (button) retryTask(button.dataset.retryId);
  });

  $("upload-btn").addEventListener("click", uploadDocument);
  $("scan-btn").addEventListener("click", scanDirectory);
  $("search-btn").addEventListener("click", runSearch);
  $("search-query").addEventListener("keydown", (event) => {
    if (event.key === "Enter") runSearch();
  });
}

async function init() {
  await loadTenants();
  const initialTab = window.location.hash.replace("#", "");
  switchTab(PAGE_TITLES[initialTab] ? initialTab : "overview", false);
}

bindEvents();
updateTenantControls();
loadReadiness();
init();
window.addEventListener("hashchange", () => {
  const tab = window.location.hash.replace("#", "");
  if (PAGE_TITLES[tab] && tab !== state.tab) switchTab(tab, false);
});
setInterval(loadReadiness, 30000);
