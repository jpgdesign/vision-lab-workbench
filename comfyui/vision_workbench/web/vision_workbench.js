import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = "http://127.0.0.1:8765";
const PRETRAINED = "原始預訓練權重";
const nodeDefinitions = new Map();
const outputViews = new Map();
let resourceData = null;
let pendingLaunch = new URL(location.href).searchParams.get("vision_workflow");

function element(tag, text, className) {
  const value = document.createElement(tag);
  if (text) value.textContent = text;
  if (className) value.className = className;
  return value;
}
function button(text, action) {
  const value = element("button", text, "vwb-button");
  value.type = "button";
  value.addEventListener("click", action);
  return value;
}
function externalLink(text, url) {
  const value = element("a", text);
  value.href = url;
  value.target = "_blank";
  value.rel = "noopener noreferrer";
  return value;
}
function resourceId(value) { return typeof value === "string" ? value.split(" | ")[0].trim() : ""; }
async function json(path) {
  const response = await fetch(path, { credentials: "same-origin" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function applyCheckpointChoices(node) {
  if (!resourceData) return;
  const model = node.widgets?.find(widget => widget.name === "model_id");
  const checkpoint = node.widgets?.find(widget => widget.name === "checkpoint_id");
  if (!model || !checkpoint) return;
  const modelId = resourceId(model.value);
  const options = resourceData.checkpoints.filter(item => item.valid && item.model_id === modelId)
    .map(item => `${item.id} | ${item.model_id} · ${item.name || item.id}`);
  const currentId = resourceId(checkpoint.value);
  const selected = options.find(value => resourceId(value) === currentId);
  checkpoint.options = { ...checkpoint.options, values: [PRETRAINED, ...options] };
  // Changing architecture invalidates its old checkpoint; never run mismatched weights.
  checkpoint.value = selected || PRETRAINED;
}

async function refreshResources() {
  resourceData = await json("/vision-workbench/resources");
  const definitions = await json("/object_info");
  for (const node of app.graph?._nodes || []) {
    if (!node.comfyClass?.startsWith("VisionWB_")) continue;
    const required = definitions[node.comfyClass]?.input?.required || {};
    for (const widget of node.widgets || []) {
      const values = required[widget.name]?.[0];
      if (!Array.isArray(values)) continue;
      const selected = values.find(value => resourceId(value) === resourceId(widget.value));
      widget.options = { ...widget.options, values };
      if (selected !== undefined) widget.value = selected;
    }
    applyCheckpointChoices(node);
  }
  app.graph?.setDirtyCanvas(true, true);
  return resourceData;
}

function resizeOutput(node, contentHeight = 90) {
  if (!node._vwbView) return;
  node._vwbView.height = contentHeight;
  node.setSize([Math.max(530, node.size?.[0] || 0), node.computeSize()[1]]);
  app.graph?.setDirtyCanvas(true, true);
}

// ComfyUI's removeInput updates link slot indexes. Keep real connected ports and
// one free slot per group; the Python signature still accepts all eight names.
function compactOutputPorts(node) {
  if (node.comfyClass !== "VisionWB_output" || node._vwbCompacting) return;
  node._vwbCompacting = true;
  try {
    const groups = ["observations", "actions", "events", "relations", "embeddings"];
    for (const group of groups) {
      let inputs = (node.inputs || []).map((input, index) => ({ input, index }))
        .filter(({ input }) => input.name === group || new RegExp(`^${group}_[2-8]$`).test(input.name));
      const empty = inputs.filter(({ input }) => input.link == null);
      for (const item of empty.slice(1).reverse()) node.removeInput(item.index);
      inputs = (node.inputs || []).filter(input => input.name === group || new RegExp(`^${group}_[2-8]$`).test(input.name));
      if (!inputs.some(input => input.link == null) && inputs.length < 8) {
        const names = new Set(inputs.map(input => input.name));
        const name = [group, ...Array.from({ length: 7 }, (_, i) => `${group}_${i + 2}`)].find(value => !names.has(value));
        node.addInput(name, group === "observations" ? "VWB_OBSERVATIONS" : "VWB_SIGNAL");
      }
    }
    resizeOutput(node, node._vwbView?.height || 90);
  } finally { node._vwbCompacting = false; }
}

function validViewer(value) {
  try {
    const url = new URL(value);
    return url.origin === BASE && url.pathname === "/" &&
      /^[A-Za-z0-9_-]{1,80}$/.test(url.searchParams.get("embedded_workflow") || "") &&
      ["camera", "vdo"].includes(url.searchParams.get("source")) ? url.href : null;
  } catch { return null; }
}

function showOutput(node, output) {
  const view = node?._vwbView;
  const details = output?.vision_workbench?.[0];
  if (!view || !details) return;
  view.content.replaceChildren();
  view.status.textContent = details.message || "流程已完成";
  const viewer = validViewer(details.viewer_url);
  if (viewer) {
    const frame = element("iframe", null, "vwb-live-frame");
    frame.title = "Vision Workbench 即時攝影機與辨識結果";
    frame.allow = "camera; autoplay; fullscreen";
    frame.allowFullscreen = true;
    frame.src = viewer;
    const enlarge = button("放大即時畫面", async () => {
      try {
        if (typeof frame.requestFullscreen !== "function") throw new Error("fullscreen-unavailable");
        await frame.requestFullscreen();
      } catch {
        view.status.textContent = "此瀏覽器目前無法放大內嵌畫面；可使用下方連結在獨立視窗開啟即時流程。";
      }
    });
    enlarge.title = "放大目前的即時畫面，不會重新載入或建立新連線；按 Esc 返回。";
    view.content.append(enlarge, externalLink("在獨立視窗開啟即時流程 ↗", viewer), frame);
    resizeOutput(node, 740);
  } else if (details.video && /^workflow_[a-f0-9]{32}\.mp4$/.test(details.video.filename)) {
    const url = `/view?${new URLSearchParams({filename: details.video.filename, subfolder: "vision-workbench", type: "output"})}`;
    const video = element("video", null, "vwb-result-video");
    video.controls = true;
    video.preload = "metadata";
    video.src = url;
    view.content.append(video, externalLink("開啟／下載結果影片 ↗", url));
    resizeOutput(node, 400);
  } else {
    view.content.append(externalLink("在 Vision Workbench 查看工作與事件 ↗", BASE));
    resizeOutput(node, 90);
  }
}

function createOutputView(node) {
  if (node._vwbView) return;
  const root = element("div", null, "vwb-output");
  const status = element("p", "影像辨識由本機 Vision Workbench 執行。按執行後，此處顯示結果；即時來源會先顯示連線面板。", "vwb-output-status");
  const content = element("div", null, "vwb-output-content");
  root.append(status, content);
  const widget = node.addDOMWidget("vision_workbench_preview", "div", root, { serialize: false, hideOnZoom: false });
  node._vwbView = { root, status, content, height: 90 };
  widget.computeSize = width => [width, node._vwbView.height];
  const previousRemoved = node.onRemoved;
  node.onRemoved = function(...args) {
    content.replaceChildren(); // Unmounting the viewer stops its camera stream.
    outputViews.delete(String(this.id));
    return previousRemoved?.apply(this, args);
  };
  const previousConnections = node.onConnectionsChange;
  node.onConnectionsChange = function(...args) {
    const result = previousConnections?.apply(this, args);
    queueMicrotask(() => compactOutputPorts(this));
    return result;
  };
  queueMicrotask(() => {
    outputViews.set(String(node.id), node);
    compactOutputPorts(node);
  });
}

async function loadTemplate(id, sourceMode = "media") {
  const data = await json("/vision-workbench/templates");
  const template = data.templates.find(value => value.id === id);
  if (!template?.workflow) throw new Error("找不到這個 Vision Workbench 範例。");
  const workflow = structuredClone(template.workflow);
  for (const node of workflow.nodes || []) {
    if (node.type === "VisionWB_source" && ["media", "camera", "vdo"].includes(sourceMode)) node.widgets_values[0] = sourceMode;
  }
  await app.loadGraphData(workflow);
  await refreshResources();
  for (const node of app.graph?._nodes || []) compactOutputPorts(node);
}

function createTemplatePanel() {
  const panel = element("section", null, "vwb-panel");
  panel.append(element("h3", "Vision Workbench 範例"), element("p", "沿用已下載模型、微調權重與素材。選擇範例和來源，載入後按 ComfyUI 的執行。"));
  const status = element("p", "正在連接本機工作檯…", "vwb-panel-status");
  const select = element("select");
  select.ariaLabel = "工作流範例";
  const description = element("p", null, "vwb-template-description");
  const source = element("select");
  source.ariaLabel = "影像來源";
  for (const [value, label] of [["media", "範例圖片／影片"], ["camera", "即時攝影機"], ["vdo", "VDO.Ninja 手機攝影機"]]) {
    const option = element("option", label); option.value = value; source.append(option);
  }
  let templates = [];
  const describe = () => { description.textContent = templates.find(value => value.id === select.value)?.description || ""; };
  select.addEventListener("change", describe);
  const load = button("載入選取範例", async () => {
    load.disabled = true;
    try { await loadTemplate(select.value, source.value); status.textContent = "範例已載入。可調整模型與權重，再按執行。"; }
    catch (error) { status.textContent = error.message; }
    finally { load.disabled = false; }
  });
  const refresh = button("重新整理範例與資源", async () => {
    refresh.disabled = true;
    try {
      const [data, resources] = await Promise.all([json("/vision-workbench/templates"), refreshResources()]);
      templates = data.templates || [];
      const current = select.value;
      select.replaceChildren(...templates.map(value => { const option = element("option", value.name); option.value = value.id; return option; }));
      if (templates.some(value => value.id === current)) select.value = current;
      describe();
      status.textContent = resources.available ? `${resources.models.length} 個模型 · ${resources.checkpoints.filter(value => value.valid).length} 組微調權重 · ${resources.media.length} 個素材` : "工作檯未啟動，請先開啟 127.0.0.1:8765。";
    } catch (error) { status.textContent = error.message; }
    finally { refresh.disabled = false; }
  });
  panel.append(status, select, description, source, load, refresh,
    element("p", "即時來源：執行後，在輸出節點內先連接攝影機，再按開始流程。攝影機與 VDO 連線資訊保留在瀏覽器，不會寫進工作流。"),
    externalLink("開啟 Vision Workbench ↗", BASE));
  refresh.click();
  return panel;
}

app.registerExtension({
  name: "VisionWorkbench.NativeBridge",
  async setup() {
    const style = element("link"); style.rel = "stylesheet"; style.href = new URL("./vision_workbench.css", import.meta.url).href; document.head.append(style);
    app.extensionManager.registerSidebarTab({id: "vision-workbench", icon: "pi pi-eye", title: "Vision Workbench", tooltip: "Vision Workbench 範例與模型", type: "custom", render: el => { el.replaceChildren(createTemplatePanel()); }});
    api.addEventListener("vision_workbench.progress", event => {
      const detail = event.detail;
      const node = app.graph?.getNodeById(detail.node);
      if (node?._vwbView) node._vwbView.status.textContent = detail.message || `工作檯執行中 · ${detail.status || "running"}`;
    });
    api.addEventListener("executed", event => {
      const node = app.graph?.getNodeById(event.detail.node);
      showOutput(node, event.detail.output);
    });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name?.startsWith("VisionWB_")) nodeDefinitions.set(nodeData.name, nodeData);
  },
  nodeCreated(node) {
    if (!node.comfyClass?.startsWith("VisionWB_")) return;
    if (node.comfyClass === "VisionWB_output") createOutputView(node);
    const model = node.widgets?.find(widget => widget.name === "model_id");
    if (model) {
      const previous = model.callback;
      model.callback = function(...args) { const result = previous?.apply(this, args); applyCheckpointChoices(node); return result; };
    }
  },
  async afterConfigureGraph() {
    for (const node of app.graph?._nodes || []) compactOutputPorts(node);
    if (pendingLaunch) {
      const id = pendingLaunch; pendingLaunch = null;
      const url = new URL(location.href);
      const source = url.searchParams.get("source") || "media";
      try {
        await loadTemplate(id, source);
        url.searchParams.delete("vision_workflow"); url.searchParams.delete("source"); history.replaceState(null, "", url);
      } catch (error) {
        app.extensionManager.toast.add({ severity: "error", summary: "Vision Workbench", detail: error.message, life: 10000 });
      }
    }
  },
});
