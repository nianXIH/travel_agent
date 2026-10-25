"use strict";

const API = "/api/v1";
const CHAT_STREAM_URL = "/api/v1/chat/stream";
const STORAGE_KEY = "userApiKey";

const STAGE_LABELS = {
  dense: "语义检索",
  sparse: "关键词检索",
  hyde: "查询扩展",
  web: "联网补充",
  fusing: "融合结果",
  rerank: "结果精排",
  generating: "生成回答",
};

const TIME_LABELS = {
  morning: "上午",
  afternoon: "下午",
  evening: "晚上",
};

const state = {
  apiKey: sessionStorage.getItem(STORAGE_KEY) || "",
  sessionId: null,
  busy: false,
  turn: 0,
  epoch: 0,
  chatAbort: null,
  activeAssistant: null,
  toastTimer: null,
};

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function setText(id, text) {
  $(id).textContent = text;
}

function safeUrl(value) {
  if (!value || typeof value !== "string") return "";
  try {
    const url = new URL(value);
    if (url.protocol === "http:" || url.protocol === "https:") return url.href;
  } catch {
    return "";
  }
  return "";
}

function clampNumber(value, min, max, fallback) {
  const number = Number.parseInt(value, 10);
  if (!Number.isFinite(number)) return fallback;
  return Math.min(max, Math.max(min, number));
}

function parseInterests(value) {
  return value
    .split(/[,，、]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 5);
}

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => node.classList.remove("show"), 2600);
}

function scrollChat() {
  const log = $("chat-log");
  log.scrollTop = log.scrollHeight;
}

/* ---------- errors and transport ---------- */

async function readError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.message === "string") return payload.message;
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail) && payload.detail.length) {
      const messages = payload.detail
        .map((item) => (item && item.msg ? String(item.msg) : ""))
        .filter(Boolean);
      if (messages.length) return messages.join("；");
    }
  } catch {
    /* non-JSON error body */
  }
  if (response.status === 403) return "当前凭证权限不足";
  if (response.status === 404) return "没有找到相关内容";
  if (response.status === 502) return "生成失败，请稍后重试";
  if (response.status === 503) return "服务暂不可用，请稍后重试";
  return `请求失败（${response.status}）`;
}

async function api(path, options = {}, epoch = state.epoch) {
  const headers = new Headers(options.headers || {});
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  if (state.apiKey) headers.set("X-API-Key", state.apiKey);

  let response;
  try {
    response = await fetch(`${API}${path}`, { ...options, headers });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("网络连接失败，请确认服务仍在运行");
  }

  if (response.status === 401) {
    if (epoch === state.epoch) handleUnauthorized();
    throw new Error("凭证无效或已失效");
  }
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

/* ---------- credential gate ---------- */

function setGateError(message) {
  const node = $("gate-error");
  node.textContent = message || "";
  node.hidden = !message;
}

function showGate(message) {
  $("workspace").hidden = true;
  $("credential-gate").hidden = false;
  setGateError(message || "");
  $("api-key").focus();
}

function resetWorkspaceView() {
  renderChatEmpty();
  setText("chat-status", "");
  $("search-results").replaceChildren();
  setText("search-status", "");
  $("plan-result").replaceChildren();
  setText("plan-status", "");
  $("recommend-result").replaceChildren();
  setText("recommend-status", "");
}

function handleUnauthorized() {
  state.apiKey = "";
  state.sessionId = null;
  state.epoch += 1;
  sessionStorage.removeItem(STORAGE_KEY);
  resetWorkspaceView();
  setConnection("未连接", "warn");
  showGate("凭证无效或已失效，请重新输入");
}

async function validateKey(key) {
  let response;
  try {
    response = await fetch(`${API}/catalog`, {
      headers: { "X-API-Key": key },
    });
  } catch {
    throw new Error("无法连接服务，请确认服务已启动");
  }
  if (response.status === 401) throw new Error("凭证无效，请检查后重试");
  if (response.status === 403) throw new Error("该凭证没有访问权限，请联系管理员");
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

function setConnection(text, tone) {
  const pill = $("connection");
  pill.textContent = text;
  pill.className = tone ? `pill ${tone}` : "pill";
}

function fillDatalist(id, values) {
  const list = $(id);
  list.replaceChildren();
  (values || []).forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    list.appendChild(option);
  });
}

function fillCatalog(catalog) {
  fillDatalist("city-options", catalog.cities);
  fillDatalist("type-options", catalog.content_types);
}

function enterWorkspace(catalog) {
  fillCatalog(catalog);
  $("credential-gate").hidden = true;
  $("workspace").hidden = false;
  setConnection(`已连接 · ${catalog.document_count || 0} 篇文档`, "ok");
  if (!$("chat-log").childElementCount) renderChatEmpty();
  $("chat-input").focus();
}

function renderChatEmpty() {
  const log = $("chat-log");
  log.replaceChildren();
  log.appendChild($("chat-empty-template").content.cloneNode(true));
}

/* ---------- tabs ---------- */

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });
  if (name === "chat") $("chat-input").focus();
}

/* ---------- chat ---------- */

function appendMessage(role, text) {
  const wrapper = el("article", `message ${role}`);
  const bubble = el("div", "bubble");
  const body = el("div", "bubble-body");
  if (text) body.textContent = text;
  bubble.appendChild(body);
  wrapper.appendChild(bubble);
  $("chat-log").appendChild(wrapper);
  scrollChat();
  return { wrapper, bubble, body };
}

function takeFrame(buffer) {
  const match = /\r?\n\r?\n/.exec(buffer);
  if (!match) return null;
  return {
    raw: buffer.slice(0, match.index),
    rest: buffer.slice(match.index + match[0].length),
  };
}

function parseSse(raw) {
  let event = "message";
  const data = [];
  raw.split(/\r?\n/).forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  });
  if (!data.length) return null;
  return { event, data: data.join("\n") };
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

function setChatBusy(busy) {
  state.busy = busy;
  const button = $("chat-send");
  button.textContent = busy ? "停止" : "发送";
  button.classList.toggle("danger", busy);
  $("chat-input").disabled = busy;
}

function stopStreaming() {
  state.turn += 1;
  if (state.activeAssistant) {
    state.activeAssistant.body.classList.remove("typing");
    if (!state.activeAssistant.received) {
      state.activeAssistant.body.textContent = "已停止生成。";
    }
    state.activeAssistant = null;
  }
  if (state.chatAbort) {
    state.chatAbort.abort();
    state.chatAbort = null;
  }
  if (state.busy) setChatBusy(false);
}

function renderChatMeta(wrapper, intent, citations) {
  const meta = el("div", "message-meta");
  if (intent) meta.appendChild(el("span", "intent-tag", `意图：${intent}`));
  if (citations.length) meta.appendChild(el("span", "", `引用 ${citations.length} 条`));
  if (meta.childElementCount) wrapper.appendChild(meta);
  if (citations.length) wrapper.appendChild(citationsDetails(citations));
}

async function sendChat() {
  const input = $("chat-input");
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  const empty = document.querySelector("#chat-log .chat-empty");
  if (empty) empty.remove();

  appendMessage("user", query);
  const assistant = appendMessage("assistant", "正在思考…");
  assistant.received = false;
  assistant.body.classList.add("typing");

  const turn = ++state.turn;
  const controller = new AbortController();
  state.chatAbort = controller;
  state.activeAssistant = assistant;
  setChatBusy(true);
  setText("chat-status", "正在检索知识库…");

  const startedAt = Date.now();
  let answer = "";
  let citations = [];
  let intent = "";
  let finished = false;

  try {
    const headers = new Headers({
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    });
    if (state.apiKey) headers.set("X-API-Key", state.apiKey);

    const response = await fetch(CHAT_STREAM_URL, {
      method: "POST",
      headers,
      body: JSON.stringify({ query, session_id: state.sessionId, top_k: 5 }),
      signal: controller.signal,
    });

    if (response.status === 401) {
      handleUnauthorized();
      throw new Error("凭证无效或已失效");
    }
    if (!response.ok) throw new Error(await readError(response));
    if (!response.body) throw new Error("当前浏览器不支持流式响应");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let frame;
      while ((frame = takeFrame(buffer)) !== null) {
        buffer = frame.rest;
        const event = parseSse(frame.raw);
        if (!event) continue;
        const payload = parseJson(event.data);

        if (event.event === "stage") {
          setText("chat-status", STAGE_LABELS[payload.stage] || "处理中…");
        } else if (event.event === "meta") {
          if (payload.session_id) state.sessionId = payload.session_id;
          if (payload.intent) intent = payload.intent;
          if (Array.isArray(payload.citations)) citations = payload.citations;
          setText("chat-status", "正在生成回答…");
        } else if (event.event === "delta") {
          answer += payload.content || "";
          assistant.received = true;
          assistant.body.classList.remove("typing");
          assistant.body.textContent = answer;
          scrollChat();
        } else if (event.event === "done") {
          finished = true;
          renderChatMeta(assistant.wrapper, intent, citations);
          const elapsed =
            typeof payload.elapsed_ms === "number" ? payload.elapsed_ms : Date.now() - startedAt;
          const usage = payload.usage || {};
          const tokens =
            typeof usage.total_tokens === "number" ? ` · ${usage.total_tokens} tokens` : "";
          setText("chat-status", `完成 · ${(elapsed / 1000).toFixed(1)}s${tokens}`);
        } else if (event.event === "error") {
          throw new Error(payload.message || "生成回答失败，请稍后重试");
        }
      }
    }

    if (state.turn !== turn) return;

    assistant.body.classList.remove("typing");
    if (!answer) {
      assistant.body.textContent = finished ? "（本次回答为空）" : "未收到回答，请稍后重试。";
    }
    if (!finished) {
      renderChatMeta(assistant.wrapper, intent, citations);
      setText("chat-status", "连接已结束");
    }
  } catch (error) {
    if (state.turn !== turn) return;
    assistant.body.classList.remove("typing");
    if (error.name === "AbortError") {
      if (!answer) assistant.body.textContent = "已停止生成。";
      setText("chat-status", "已停止生成");
    } else {
      assistant.body.textContent = answer || "";
      assistant.wrapper.appendChild(
        el("p", "error", error.message || "生成回答失败，请稍后重试")
      );
      setText("chat-status", "");
    }
  } finally {
    if (state.turn === turn) {
      state.chatAbort = null;
      state.activeAssistant = null;
      setChatBusy(false);
      scrollChat();
    }
  }
}

/* ---------- citations ---------- */

function citationTitle(citation) {
  return (
    citation.title ||
    citation.scenic_name ||
    citation.hotel_name ||
    citation.restaurant_name ||
    citation.route_name ||
    citation.source_file ||
    citation.chunk_id ||
    "来源片段"
  );
}

function citationsDetails(citations) {
  const details = el("details", "citations");
  details.appendChild(el("summary", "", `引用来源（${citations.length}）`));
  const list = el("ol", "citation-list");
  citations.forEach((citation) => {
    const item = el("li");
    item.appendChild(el("strong", "", citationTitle(citation)));
    const meta = [citation.city, citation.content_type, citation.section_path].filter(Boolean);
    if (meta.length) item.appendChild(el("span", "muted", ` · ${meta.join(" · ")}`));
    const url = safeUrl(citation.url);
    if (url) {
      item.appendChild(document.createTextNode(" "));
      const link = el("a", "", "查看原文 ↗");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      item.appendChild(link);
    }
    list.appendChild(item);
  });
  details.appendChild(list);
  return details;
}

function citationCard(citation) {
  const card = el("article", "card citation-card");
  card.appendChild(el("h3", "", citationTitle(citation)));
  const meta = [citation.city, citation.content_type, citation.source_file].filter(Boolean);
  if (meta.length) card.appendChild(el("p", "muted small", meta.join(" · ")));
  if (typeof citation.score === "number") {
    card.appendChild(el("span", "pill", `相关度 ${citation.score.toFixed(3)}`));
  }
  card.appendChild(el("p", "citation-body", citation.content || "（空片段）"));
  const url = safeUrl(citation.url);
  if (url) {
    const link = el("a", "small", "查看原文 ↗");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    card.appendChild(link);
  }
  return card;
}

/* ---------- search ---------- */

function setFormBusy(buttonId, busy, label) {
  const button = $(buttonId);
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = "处理中…";
    button.disabled = true;
  } else {
    button.textContent = label || button.dataset.label || "提交";
    button.disabled = false;
  }
}

async function runSearch(event) {
  event.preventDefault();
  if ($("search-submit").disabled) return;
  const epoch = state.epoch;

  const query = $("search-query").value.trim();
  if (!query) {
    toast("请输入检索内容");
    return;
  }

  const payload = {
    query,
    mode: $("search-mode").value,
    top_k: clampNumber($("search-topk").value, 1, 20, 5),
  };
  const city = $("search-city").value.trim();
  const type = $("search-type").value.trim();
  if (city) payload.city = city;
  if (type) payload.content_type = type;

  setFormBusy("search-submit", true);
  setText("search-status", "检索中…");
  $("search-results").replaceChildren();

  try {
    const data = await api(
      "/search",
      { method: "POST", body: JSON.stringify(payload) },
      epoch
    );
    if (epoch !== state.epoch) return;
    const citations = Array.isArray(data.citations) ? data.citations : [];
    if (!citations.length) {
      setText("search-status", "没有找到匹配内容，换个关键词或去掉筛选条件试试。");
      return;
    }
    setText("search-status", `找到 ${citations.length} 条结果 · 用时 ${data.elapsed_ms} ms`);
    const results = $("search-results");
    citations.forEach((citation) => results.appendChild(citationCard(citation)));
  } catch (error) {
    if (epoch !== state.epoch) return;
    setText("search-status", "");
    $("search-results").replaceChildren(el("p", "error", error.message));
  } finally {
    setFormBusy("search-submit", false, "检索");
  }
}

/* ---------- itinerary ---------- */

function budgetItem(label, value, total) {
  const item = el("div", total ? "budget-item total" : "budget-item");
  item.appendChild(el("span", "label", label));
  item.appendChild(el("span", "value", `${value} 元`));
  return item;
}

function budgetCard(budget) {
  const card = el("div", "card");
  card.appendChild(el("h2", "", "预算参考"));
  const grid = el("div", "budget-grid");
  [
    ["transport", "交通"],
    ["accommodation", "住宿"],
    ["food", "餐饮"],
    ["tickets", "门票"],
    ["other", "其他"],
  ].forEach(([key, label]) => {
    const value = Number(budget[key] || 0);
    if (value) grid.appendChild(budgetItem(label, value));
  });
  grid.appendChild(budgetItem("合计", Number(budget.total || 0), true));
  card.appendChild(grid);
  card.appendChild(el("p", "muted small", "预算为模型估算，实际以商家报价为准。"));
  return card;
}

function listCard(title, items) {
  const card = el("div", "card");
  card.appendChild(el("h2", "", title));
  const list = el("ul", "bullet-list");
  items.forEach((item) => list.appendChild(el("li", "", item)));
  card.appendChild(list);
  return card;
}

function renderPlan(data) {
  const container = $("plan-result");
  const plan = data.plan || {};

  const head = el("div", "card result-head");
  head.appendChild(
    el("h2", "", `${data.resolved_city || data.destination || "目的地"} · ${data.days} 天行程`)
  );
  if (plan.summary) head.appendChild(el("p", "muted", plan.summary));
  container.appendChild(head);

  (Array.isArray(plan.days) ? plan.days : []).forEach((day) => {
    const card = el("article", "card day-card");
    const dayHead = el("div", "day-head");
    dayHead.appendChild(el("span", "day-index", `第 ${day.day} 天`));
    if (day.theme) dayHead.appendChild(el("h3", "", day.theme));
    card.appendChild(dayHead);

    if (day.area) card.appendChild(el("p", "muted small", `区域：${day.area}`));

    const stops = el("ol", "stops");
    (Array.isArray(day.stops) ? day.stops : []).forEach((stop) => {
      const item = el("li", "stop");
      item.appendChild(
        el("span", "time-tag", TIME_LABELS[stop.time] || stop.time || "")
      );
      const body = el("div", "stop-body");
      body.appendChild(el("strong", "", stop.title || ""));
      if (stop.activity) body.appendChild(el("p", "", stop.activity));
      if (stop.duration_hours) {
        body.appendChild(el("p", "muted small", `建议停留 ${stop.duration_hours} 小时`));
      }
      if (stop.tips) body.appendChild(el("p", "tip", stop.tips));
      item.appendChild(body);
      stops.appendChild(item);
    });
    if (stops.childElementCount) card.appendChild(stops);

    if (Array.isArray(day.meals) && day.meals.length) {
      card.appendChild(el("p", "muted small", `餐饮：${day.meals.join(" · ")}`));
    }
    if (day.hotel) card.appendChild(el("p", "muted small", `住宿：${day.hotel}`));
    if (day.budget_estimate) {
      card.appendChild(el("p", "muted small", `当日预算约 ${day.budget_estimate} 元`));
    }
    container.appendChild(card);
  });

  if (plan.budget) container.appendChild(budgetCard(plan.budget));
  if (Array.isArray(plan.tips) && plan.tips.length) {
    container.appendChild(listCard("实用提示", plan.tips));
  }
  if (Array.isArray(plan.alternatives) && plan.alternatives.length) {
    container.appendChild(listCard("备选方案", plan.alternatives));
  }
  if (Array.isArray(data.citations) && data.citations.length) {
    container.appendChild(citationsDetails(data.citations));
  }
}

async function runPlan(event) {
  event.preventDefault();
  if ($("plan-submit").disabled) return;
  const epoch = state.epoch;

  const destination = $("plan-destination").value.trim();
  if (!destination) {
    toast("请填写目的地");
    return;
  }

  const payload = {
    destination,
    days: clampNumber($("plan-days").value, 1, 10, 3),
    travelers: clampNumber($("plan-travelers").value, 1, 20, 1),
    traveler_type: $("plan-type").value,
    pace: $("plan-pace").value,
    interests: parseInterests($("plan-interests").value),
  };
  const budget = Number.parseInt($("plan-budget").value, 10);
  if (Number.isFinite(budget) && budget >= 0) payload.budget = budget;
  const hotel = $("plan-hotel").value.trim();
  if (hotel) payload.hotel_preference = hotel;
  const notes = $("plan-notes").value.trim();
  if (notes) payload.notes = notes;

  setFormBusy("plan-submit", true);
  setText("plan-status", "正在检索资料并生成行程…");
  $("plan-result").replaceChildren();

  try {
    const data = await api(
      "/itineraries",
      { method: "POST", body: JSON.stringify(payload) },
      epoch
    );
    if (epoch !== state.epoch) return;
    setText("plan-status", `生成完成 · 用时 ${(data.elapsed_ms / 1000).toFixed(1)}s`);
    renderPlan(data);
  } catch (error) {
    if (epoch !== state.epoch) return;
    setText("plan-status", "");
    $("plan-result").replaceChildren(el("p", "error", error.message));
  } finally {
    setFormBusy("plan-submit", false, "生成行程");
  }
}

/* ---------- recommendations ---------- */

function recommendationCard(item) {
  const card = el("article", "card rec-card");
  const titleRow = el("div", "chips");
  titleRow.appendChild(el("h3", "", item.name || "未命名推荐"));
  if (item.category) titleRow.appendChild(el("span", "chip rec-category", item.category));
  card.appendChild(titleRow);
  if (item.reason) card.appendChild(el("p", "", item.reason));
  if (Array.isArray(item.highlights) && item.highlights.length) {
    const chips = el("div", "chips");
    item.highlights.forEach((text) => chips.appendChild(el("span", "chip", text)));
    card.appendChild(chips);
  }
  if (item.best_for) card.appendChild(el("p", "muted small", `适合：${item.best_for}`));
  if (Array.isArray(item.sources) && item.sources.length) {
    card.appendChild(el("p", "muted small", `来源编号：${item.sources.join("、")}`));
  }
  return card;
}

function renderRecommendations(data) {
  const container = $("recommend-result");
  const set = data.recommendations || {};

  const head = el("div", "card result-head");
  head.appendChild(
    el(
      "h2",
      "",
      `${data.resolved_city || data.destination || "目的地"} · ${data.category || "综合"}推荐`
    )
  );
  if (set.summary) head.appendChild(el("p", "muted", set.summary));
  container.appendChild(head);

  const items = Array.isArray(set.items) ? set.items : [];
  if (items.length) {
    const grid = el("div", "rec-grid");
    items.forEach((item) => grid.appendChild(recommendationCard(item)));
    container.appendChild(grid);
  }
  if (Array.isArray(set.tips) && set.tips.length) {
    container.appendChild(listCard("小贴士", set.tips));
  }
  if (Array.isArray(data.citations) && data.citations.length) {
    container.appendChild(citationsDetails(data.citations));
  }
}

async function runRecommend(event) {
  event.preventDefault();
  if ($("recommend-submit").disabled) return;
  const epoch = state.epoch;

  const destination = $("recommend-destination").value.trim();
  if (!destination) {
    toast("请填写目的地");
    return;
  }

  const payload = {
    destination,
    category: $("recommend-category").value,
    top_k: clampNumber($("recommend-topk").value, 1, 10, 5),
    traveler_type: $("recommend-type").value,
    interests: parseInterests($("recommend-interests").value),
  };
  const notes = $("recommend-notes").value.trim();
  if (notes) payload.notes = notes;

  setFormBusy("recommend-submit", true);
  setText("recommend-status", "正在检索资料并整理推荐…");
  $("recommend-result").replaceChildren();

  try {
    const data = await api(
      "/recommendations",
      { method: "POST", body: JSON.stringify(payload) },
      epoch
    );
    if (epoch !== state.epoch) return;
    setText(
      "recommend-status",
      `生成完成 · 用时 ${(data.elapsed_ms / 1000).toFixed(1)}s`
    );
    renderRecommendations(data);
  } catch (error) {
    if (epoch !== state.epoch) return;
    setText("recommend-status", "");
    $("recommend-result").replaceChildren(el("p", "error", error.message));
  } finally {
    setFormBusy("recommend-submit", false, "获取推荐");
  }
}

/* ---------- wiring ---------- */

function bindEvents() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
  });

  $("gate-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const key = $("api-key").value.trim();
    if (!key) {
      setGateError("请输入访问凭证");
      return;
    }

    const button = $("gate-submit");
    button.disabled = true;
    button.textContent = "验证中…";
    setGateError("");

    try {
      const catalog = await validateKey(key);
      state.apiKey = key;
      sessionStorage.setItem(STORAGE_KEY, key);
      $("api-key").value = "";
      enterWorkspace(catalog);
    } catch (error) {
      setGateError(error.message || "验证失败，请稍后重试");
    } finally {
      button.disabled = false;
      button.textContent = "进入";
    }
  });

  $("change-key").addEventListener("click", () => {
    stopStreaming();
    state.apiKey = "";
    state.sessionId = null;
    state.epoch += 1;
    sessionStorage.removeItem(STORAGE_KEY);
    resetWorkspaceView();
    setConnection("未连接");
    showGate("");
  });

  $("new-chat").addEventListener("click", () => {
    stopStreaming();
    state.sessionId = null;
    renderChatEmpty();
    setText("chat-status", "");
    $("chat-input").focus();
  });

  $("chat-log").addEventListener("click", (event) => {
    const target = event.target.closest("[data-prompt]");
    if (!target || state.busy) return;
    $("chat-input").value = target.dataset.prompt || "";
    $("chat-form").requestSubmit();
  });

  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.busy) {
      stopStreaming();
      setText("chat-status", "已停止生成");
      return;
    }
    sendChat();
  });

  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  $("search-form").addEventListener("submit", runSearch);
  $("plan-form").addEventListener("submit", runPlan);
  $("recommend-form").addEventListener("submit", runRecommend);
}

async function bootstrap() {
  bindEvents();
  if (!state.apiKey) {
    $("api-key").focus();
    return;
  }
  try {
    const catalog = await validateKey(state.apiKey);
    enterWorkspace(catalog);
  } catch {
    state.apiKey = "";
    sessionStorage.removeItem(STORAGE_KEY);
    showGate("之前的凭证已失效，请重新输入");
  }
}

bootstrap();
