const chatForm = document.getElementById("chatForm");
const queryInput = document.getElementById("queryInput");
const chatFeed = document.getElementById("chatFeed");
const emptyState = document.getElementById("emptyState");
const insightContent = document.getElementById("insightContent");
const assistantSummary = document.getElementById("assistantSummary");
const intentValue = document.getElementById("intentValue");
const toolValue = document.getElementById("toolValue");
const apiStatus = document.getElementById("apiStatus");
const promptCards = document.querySelectorAll(".prompt-card");
const promptSuggestions = document.getElementById("promptSuggestions");

const forecastView = document.getElementById("forecastView");
const compareView = document.getElementById("compareView");
const alertsView = document.getElementById("alertsView");

const entityTitle = document.getElementById("entityTitle");
const entityMicro = document.getElementById("entityMicro");
const clusterText = document.getElementById("clusterText");
const baselineValue = document.getElementById("baselineValue");
const changeValue = document.getElementById("changeValue");
const totalForecastValue = document.getElementById("totalForecastValue");
const historyWindowChip = document.getElementById("historyWindowChip");
const timelineChip = document.getElementById("timelineChip");
const timelineTableBody = document.getElementById("timelineTableBody");
const forecastChart = document.getElementById("forecastChart");
const chartTooltip = document.getElementById("chartTooltip");
const historyPanel = document.getElementById("historyPanel");
const historySummary = document.getElementById("historySummary");
const qualityPanel = document.getElementById("qualityPanel");
const qualitySummary = document.getElementById("qualitySummary");

const comparisonSummary = document.getElementById("comparisonSummary");
const comparisonCards = document.getElementById("comparisonCards");

const alertSummary = document.getElementById("alertSummary");
const alertsTableBody = document.getElementById("alertsTableBody");

addAssistantMessage(
  "Ask for a product `stock_code` or customer `customer_id`. I will route the request through the agent layer and show the latest trained forecast context."
);
checkHealth();

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }

  hidePromptSuggestions();
  addUserMessage(query);
  queryInput.value = "";
  const loadingNode = addLoadingMessage();

  try {
    const response = await fetch("/api/v1/agent/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        default_entity_type: null,
        horizon_weeks: 4,
        target_metric: "revenue",
      }),
    });

    const body = await response.json();
    loadingNode.remove();

    if (!response.ok) {
      setOfflineStatus();
      addAssistantMessage(body.detail || "The request failed.");
      renderError(body.detail || "The request failed.");
      return;
    }

    setOnlineStatus();
    addAssistantMessage(body.summary, `${body.intent} · ${body.tool_used}`);
    renderResponse(body);
  } catch (error) {
    loadingNode.remove();
    setOfflineStatus();
    const message = error instanceof Error ? error.message : "Network error";
    addAssistantMessage(message);
    renderError(message);
  }
});

promptCards.forEach((card) => {
  card.addEventListener("click", () => {
    queryInput.value = card.dataset.prompt || "";
    queryInput.focus();
  });
});

function hidePromptSuggestions() {
  if (!promptSuggestions) {
    return;
  }
  promptSuggestions.classList.add("hidden");
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed");
    }
    setOnlineStatus();
  } catch {
    setOfflineStatus();
  }
}

function renderResponse(agentResponse) {
  emptyState.classList.add("hidden");
  insightContent.classList.remove("hidden");
  hideAllViews();

  assistantSummary.textContent = agentResponse.summary || "No summary returned.";
  intentValue.textContent = prettify(agentResponse.intent);
  toolValue.textContent = agentResponse.tool_used || "-";

  if (agentResponse.intent === "compare") {
    renderComparison(agentResponse.payload);
    return;
  }

  if (agentResponse.intent === "alerts") {
    renderAlerts(agentResponse.payload);
    return;
  }

  renderForecast(agentResponse.payload);
  void loadEntityContext(agentResponse.payload);
  void loadHistory(agentResponse.payload);
}

function renderForecast(payload) {
  forecastView.classList.remove("hidden");
  entityTitle.textContent = `${payload.entity_type} ${payload.entity_id}`;
  entityMicro.textContent = "";
  entityMicro.title = "";
  entityMicro.classList.add("hidden");
  clusterText.textContent = `${payload.cluster_label} · ${payload.cluster_id}`;
  baselineValue.textContent = formatMetric(payload.baseline_recent_average, payload.target_metric);
  changeValue.textContent = formatChange(payload.first_week_change_pct);
  totalForecastValue.textContent = formatMetric(payload.total_forecast, payload.target_metric);
  historyWindowChip.textContent = "Loading actuals";
  timelineChip.textContent = "Forecast only";
  historyPanel.classList.add("hidden");
  historySummary.textContent = "";

  if (payload.data_quality_summary) {
    qualityPanel.classList.remove("hidden");
    qualitySummary.textContent = payload.data_quality_summary;
  } else {
    qualityPanel.classList.add("hidden");
    qualitySummary.textContent = "";
  }

  timelineTableBody.innerHTML = "";
  (payload.forecast || []).forEach((point) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${point.week_start}</td>
      <td>Forecast</td>
      <td>${formatMetric(point.value, payload.target_metric)}</td>
    `;
    timelineTableBody.appendChild(row);
  });

  drawTimelineChart([], payload.forecast || [], payload.target_metric);
}

async function loadEntityContext(payload) {
  try {
    const params = new URLSearchParams({
      entity_type: payload.entity_type,
      entity_id: payload.entity_id,
    });
    const response = await fetch(`/api/v1/context/entity?${params.toString()}`);
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Could not load entity context.");
    }

    const descriptor = body.short_label || body.note || "";
    if (!descriptor) {
      entityMicro.textContent = "";
      entityMicro.title = "";
      entityMicro.classList.add("hidden");
      return;
    }

    entityMicro.textContent = descriptor;
    entityMicro.title = descriptor;
    entityMicro.classList.remove("hidden");
  } catch {
    entityMicro.textContent = "";
    entityMicro.title = "";
    entityMicro.classList.add("hidden");
  }
}

function renderComparison(payload) {
  compareView.classList.remove("hidden");
  comparisonSummary.textContent = payload.comparison_summary || "No comparison summary returned.";
  comparisonCards.innerHTML = "";

  (payload.forecasts || []).forEach((forecast) => {
    const card = document.createElement("article");
    card.className = "panel comparison-card";
    card.innerHTML = `
      <div>
        <p class="panel-label">${forecast.entity_type}</p>
        <strong>${forecast.entity_id}</strong>
        <p class="meta">${forecast.cluster_label}</p>
      </div>
      <div>
        <div class="metric-pair">
          <span>Recent avg</span>
          <strong>${formatMetric(forecast.baseline_recent_average, forecast.target_metric)}</strong>
        </div>
        <div class="metric-pair">
          <span>Total forecast</span>
          <strong>${formatMetric(forecast.total_forecast, forecast.target_metric)}</strong>
        </div>
      </div>
    `;
    comparisonCards.appendChild(card);
  });
}

function renderAlerts(payload) {
  alertsView.classList.remove("hidden");
  alertSummary.textContent = payload.summary || "No alert summary returned.";
  alertsTableBody.innerHTML = "";

  (payload.alerts || []).forEach((alert) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${alert.entity_id}</td>
      <td>${alert.cluster_label}</td>
      <td>${formatNumber(alert.recent_average)}</td>
      <td>${formatNumber(alert.forecast_average)}</td>
      <td>${(alert.decline_pct * 100).toFixed(1)}%</td>
    `;
    alertsTableBody.appendChild(row);
  });
}

function renderError(message) {
  emptyState.classList.add("hidden");
  insightContent.classList.remove("hidden");
  hideAllViews();
  assistantSummary.textContent = message;
  intentValue.textContent = "Error";
  toolValue.textContent = "-";
}

function hideAllViews() {
  forecastView.classList.add("hidden");
  compareView.classList.add("hidden");
  alertsView.classList.add("hidden");
  qualityPanel.classList.add("hidden");
  historyPanel.classList.add("hidden");
  comparisonCards.innerHTML = "";
  alertsTableBody.innerHTML = "";
  timelineTableBody.innerHTML = "";
  forecastChart.innerHTML = "";
  hideChartTooltip();
}

function addUserMessage(text) {
  appendMessage("user", text, timestampLabel());
}

function addAssistantMessage(text, meta = timestampLabel()) {
  appendMessage("assistant", text, meta);
}

function addLoadingMessage() {
  const wrapper = document.createElement("article");
  wrapper.className = "message assistant";
  wrapper.innerHTML = `
    <div class="message-bubble">
      <div class="loading-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
    <div class="message-meta">Routing request</div>
  `;
  chatFeed.appendChild(wrapper);
  chatFeed.scrollTop = chatFeed.scrollHeight;
  return wrapper;
}

function appendMessage(role, text, meta) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <div class="message-bubble">${escapeHtml(text)}</div>
    <div class="message-meta">${escapeHtml(meta)}</div>
  `;
  chatFeed.appendChild(wrapper);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

async function loadHistory(payload) {
  const lookbackWeeks = 4;
  try {
    const params = new URLSearchParams({
      entity_type: payload.entity_type,
      entity_id: payload.entity_id,
      target_metric: payload.target_metric,
      lookback_weeks: String(lookbackWeeks),
    });
    const response = await fetch(`/api/v1/history/entity?${params.toString()}`);
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Could not load actual history.");
    }
    renderTimeline(body, payload);
  } catch (error) {
    historyWindowChip.textContent = "Actuals unavailable";
    timelineChip.textContent = "Forecast only";
    historyPanel.classList.remove("hidden");
    historySummary.textContent =
      error instanceof Error ? error.message : "Could not load actual history for this entity.";
    drawTimelineChart([], payload.forecast || [], payload.target_metric);
  }
}

function renderTimeline(historyPayload, forecastPayload) {
  const actuals = historyPayload.points || [];
  const forecast = forecastPayload.forecast || [];
  historyWindowChip.textContent = actuals.length
    ? `${actuals.length} actual weeks`
    : "No clean actuals";
  timelineChip.textContent = `${forecast.length} forecast weeks`;

  if (historyPayload.summary) {
    historyPanel.classList.remove("hidden");
    historySummary.textContent = historyPayload.summary;
  } else {
    historyPanel.classList.add("hidden");
    historySummary.textContent = "";
  }

  timelineTableBody.innerHTML = "";
  [...actuals, ...forecast.map((point) => ({ ...point, source: "forecast" }))].forEach((point) => {
    const source = point.source || "actual";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${point.week_start}</td>
      <td>${prettify(source)}</td>
      <td>${formatMetric(point.value, forecastPayload.target_metric)}</td>
    `;
    timelineTableBody.appendChild(row);
  });

  drawTimelineChart(actuals, forecast, forecastPayload.target_metric);
}

function drawTimelineChart(actualPoints, forecastPoints, metric) {
  forecastChart.innerHTML = "";
  hideChartTooltip();
  const actuals = actualPoints || [];
  const forecast = forecastPoints || [];
  const combined = [
    ...actuals.map((point) => ({ ...point, source: "actual" })),
    ...forecast.map((point) => ({ ...point, source: "forecast" })),
  ];
  if (!combined.length) {
    return;
  }

  const width = 640;
  const height = 260;
  const padding = { top: 24, right: 24, bottom: 32, left: 46 };
  const values = combined.map((point) => Number(point.value || 0));
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const xStep = (width - padding.left - padding.right) / Math.max(combined.length - 1, 1);
  const yScale = (value) => {
    const span = max - min || 1;
    return height - padding.bottom - ((value - min) / span) * (height - padding.top - padding.bottom);
  };

  const buildPath = (points, offset = 0) =>
    points
    .map((point, index) => {
      const x = padding.left + (index + offset) * xStep;
      const y = yScale(Number(point.value || 0));
      return `${index === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const actualPath = actuals.length ? buildPath(actuals) : "";
  const forecastPath = forecast.length ? buildPath(forecast, actuals.length ? actuals.length - 1 : 0) : "";
  const forecastBoundaryIndex = actuals.length ? actuals.length - 1 : 0;
  const boundaryX = padding.left + forecastBoundaryIndex * xStep;
  const firstForecastX = padding.left + (actuals.length ? actuals.length - 1 : 0) * xStep;
  const areaPath = forecast.length
    ? `${forecastPath} L ${padding.left + (combined.length - 1) * xStep} ${height - padding.bottom} L ${firstForecastX} ${height - padding.bottom} Z`
    : "";

  forecastChart.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="#fffdf7"></rect>
    <g stroke="rgba(18, 32, 38, 0.08)" stroke-width="1">
      ${[0, 0.25, 0.5, 0.75, 1]
        .map((ratio) => {
          const y = padding.top + ratio * (height - padding.top - padding.bottom);
          return `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>`;
        })
        .join("")}
    </g>
    ${actuals.length ? `<path d="${actualPath}" fill="none" stroke="#d9a441" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
    ${forecast.length ? `<path d="${areaPath}" fill="rgba(31, 124, 127, 0.12)"></path>` : ""}
    ${forecast.length ? `<path d="${forecastPath}" fill="none" stroke="#1f7c7f" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
    ${forecast.length && actuals.length ? `<line x1="${boundaryX}" y1="${padding.top}" x2="${boundaryX}" y2="${height - padding.bottom}" stroke="rgba(18, 32, 38, 0.18)" stroke-dasharray="6 5"></line>` : ""}
    ${combined
      .map((point, index) => {
        const x = padding.left + index * xStep;
        const y = yScale(Number(point.value || 0));
        const fill = point.source === "actual" ? "#d9a441" : "#bb5c31";
        const label = point.source === "actual" ? "Actual" : "Forecast";
        return `
          <g>
            <circle cx="${x}" cy="${y}" r="4.5" fill="${fill}"></circle>
            <circle
              class="chart-hit"
              cx="${x}"
              cy="${y}"
              r="13"
              fill="transparent"
              data-week="${point.week_start}"
              data-value="${Number(point.value || 0)}"
              data-source="${point.source}"
              data-label="${label}"
              data-x="${x}"
              data-y="${y}"
            >
              <title>${label} · ${point.week_start} · ${formatMetric(point.value, metric)}</title>
            </circle>
          </g>
        `;
      })
      .join("")}
    <text x="${padding.left}" y="${height - 10}" fill="#57666d" font-size="12">${combined[0].week_start}</text>
    <text x="${width - padding.right - 70}" y="${height - 10}" fill="#57666d" font-size="12">${combined[combined.length - 1].week_start}</text>
    ${actuals.length ? `<text x="${width - padding.right - 152}" y="${padding.top + 8}" fill="#d9a441" font-size="12">Actuals</text>` : ""}
    ${forecast.length ? `<text x="${width - padding.right - 74}" y="${padding.top + 8}" fill="#1f7c7f" font-size="12">Forecast</text>` : ""}
  `;

  bindChartTooltip(metric);
}

function bindChartTooltip(metric) {
  forecastChart.querySelectorAll(".chart-hit").forEach((node) => {
    node.addEventListener("mouseenter", () => showChartTooltip(node, metric));
    node.addEventListener("mousemove", () => showChartTooltip(node, metric));
    node.addEventListener("mouseleave", hideChartTooltip);
  });
}

function showChartTooltip(node, metric) {
  const chartPanel = forecastChart.closest(".chart-panel");
  if (!chartPanel || !chartTooltip) {
    return;
  }

  chartTooltip.innerHTML = `
    <strong>${escapeHtml(node.dataset.label || "-")}</strong>
    <span>${escapeHtml(node.dataset.week || "-")}</span>
    <span>${escapeHtml(formatMetric(Number(node.dataset.value || 0), metric))}</span>
  `;
  chartTooltip.classList.remove("hidden");

  const chartPanelRect = chartPanel.getBoundingClientRect();
  const pointRect = node.getBoundingClientRect();
  const rawLeft = pointRect.left - chartPanelRect.left + pointRect.width / 2;
  const rawTop = pointRect.top - chartPanelRect.top + pointRect.height / 2;
  const maxLeft = chartPanel.clientWidth - chartTooltip.offsetWidth / 2 - 12;
  const minLeft = chartTooltip.offsetWidth / 2 + 12;
  const clampedLeft = Math.min(Math.max(rawLeft, minLeft), maxLeft);
  const clampedTop = Math.max(rawTop, chartTooltip.offsetHeight + 20);

  chartTooltip.style.left = `${clampedLeft}px`;
  chartTooltip.style.top = `${clampedTop}px`;
}

function hideChartTooltip() {
  if (!chartTooltip) {
    return;
  }
  chartTooltip.classList.add("hidden");
}

function setOnlineStatus() {
  apiStatus.textContent = "API online";
  apiStatus.classList.add("online");
}

function setOfflineStatus() {
  apiStatus.textContent = "API issue";
  apiStatus.classList.remove("online");
}

function timestampLabel() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatMetric(value, metric) {
  const rendered = formatNumber(value);
  return metric === "quantity" ? `${rendered} units` : rendered;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  });
}

function formatChange(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "No baseline";
  }
  const numeric = Number(value);
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(1)}%`;
}

function prettify(value) {
  return String(value || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
}
