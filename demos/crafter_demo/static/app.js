// Crafter Survival Demo — WebSocket client + UI state management

const WS_URL = `ws://${location.host}/ws`;
const RECONNECT_DELAY = 2000;

let ws = null;
let connected = false;
let goals = [];
let logBuffer = [];
const MAX_LOG = 100;

// --- WebSocket ---

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    connected = true;
    console.log('WS connected');
  };

  ws.onclose = () => {
    connected = false;
    console.log('WS closed, reconnecting...');
    setTimeout(connect, RECONNECT_DELAY);
  };

  ws.onerror = (e) => {
    console.error('WS error', e);
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'frame') {
      handleFrame(msg);
    } else if (msg.type === 'train_progress') {
      handleTrainProgress(msg);
    }
  };
}

function send(cmd) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(cmd));
  }
}

// --- Frame handler ---

const gameCanvas = document.getElementById('gameCanvas');
const gameCtx = gameCanvas.getContext('2d');
const minimapCanvas = document.getElementById('minimapCanvas');
const minimapCtx = minimapCanvas.getContext('2d');

function handleFrame(f) {
  // Game frame
  if (f.frame) {
    const img = new Image();
    img.onload = () => {
      gameCtx.imageSmoothingEnabled = false;
      gameCtx.drawImage(img, 0, 0, 64, 64);
    };
    img.src = 'data:image/png;base64,' + f.frame;
  }

  // Minimap
  if (f.minimap) {
    const mimg = new Image();
    mimg.onload = () => {
      minimapCtx.imageSmoothingEnabled = false;
      minimapCtx.drawImage(mimg, 0, 0, 9, 9);
    };
    mimg.src = 'data:image/png;base64,' + f.minimap;
  }

  // Episode / step
  document.getElementById('epNum').textContent = f.episode || 0;
  document.getElementById('stepNum').textContent = f.step || 0;

  // Survival bars
  updateBar('Health', f.survival?.health ?? 9);
  updateBar('Food', f.survival?.food ?? 9);
  updateBar('Drink', f.survival?.drink ?? 9);
  updateBar('Energy', f.survival?.energy ?? 9);

  // Inventory
  updateInventory(f.inventory || {});

  // Near detection
  const nearBadge = document.getElementById('nearBadge');
  nearBadge.textContent = f.agent?.near || 'empty';

  // Agent info
  const agentInfo = document.getElementById('agentInfo');
  const a = f.agent || {};
  agentInfo.textContent = `${a.action || 'noop'} (${a.reason || 'idle'})`;

  // Reactive banner
  const banner = document.getElementById('reactiveBanner');
  if (f.reactive && f.reactive.action) {
    banner.className = 'reactive-banner ' + (f.reactive.reason === 'danger' ? 'danger' : 'survival');
    const target = f.reactive.target ? ` ${f.reactive.target}` : '';
    banner.textContent = `${f.reactive.reason}: ${f.reactive.action}${target}`;
  } else {
    banner.className = 'reactive-banner hidden';
  }

  // Plan
  updatePlan(f.plan || []);

  // Causal graph
  if (f.confidence && typeof renderCausalGraph === 'function') {
    renderCausalGraph(f.confidence);
  }

  // Metrics
  if (f.metrics?.history && typeof updateSparklines === 'function') {
    updateSparklines(f.metrics.history);
  }

  // Log
  if (f.log && f.log.length > 0) {
    for (const line of f.log) {
      logBuffer.push(line);
      if (logBuffer.length > MAX_LOG) logBuffer.shift();
    }
    updateLogDisplay();
  }

  // No-model badge
  const badge = document.getElementById('noModelBadge');
  // We check via status endpoint on load; frame doesn't carry has_model
}

function updateBar(name, value) {
  const max = 9;
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  document.getElementById('bar' + name).style.width = pct + '%';
  document.getElementById('val' + name).textContent = value;
}

function updateInventory(inv) {
  const grid = document.getElementById('invGrid');
  grid.innerHTML = '';
  for (const [k, v] of Object.entries(inv)) {
    if (v > 0) {
      const el = document.createElement('div');
      el.className = 'inv-item';
      el.innerHTML = `${k}: <span class="count">${v}</span>`;
      grid.appendChild(el);
    }
  }
}

function updatePlan(plan) {
  const list = document.getElementById('planList');
  list.innerHTML = '';
  for (const step of plan) {
    const el = document.createElement('div');
    el.className = 'plan-step ' + step.status;
    const icon = step.status === 'done' ? '\u2713' : step.status === 'active' ? '\u25B6' : '\u25CB';
    el.innerHTML = `<span class="icon">${icon}</span> ${step.label}`;
    list.appendChild(el);
  }
}

function updateLogDisplay() {
  const container = document.getElementById('logEntries');
  // Show last 8 entries
  const recent = logBuffer.slice(-8);
  container.innerHTML = recent.map(l => `<span class="log-entry">${l}</span>`).join(' &nbsp; ');
}

// --- Train progress ---

function handleTrainProgress(tp) {
  const fill = document.getElementById('trainProgress');
  const status = document.getElementById('trainStatus');

  if (tp.done) {
    fill.style.width = '100%';
    if (tp.error) {
      status.textContent = `Error: ${tp.error}`;
    } else {
      status.textContent = 'Training complete! Model loaded.';
      setTimeout(() => {
        document.getElementById('trainModal').classList.remove('open');
        document.getElementById('noModelBadge').style.display = 'none';
      }, 2000);
    }
    return;
  }

  const pct = tp.total_epochs > 0 ? (tp.epoch / tp.total_epochs * 100) : 0;
  fill.style.width = pct + '%';
  status.textContent = `${tp.phase}: ${tp.epoch}/${tp.total_epochs} (loss: ${tp.loss.toFixed(4)})`;
}

// --- Controls ---

document.getElementById('btnPlay').addEventListener('click', () => send({cmd: 'play'}));
document.getElementById('btnPause').addEventListener('click', () => send({cmd: 'pause'}));
document.getElementById('btnStep').addEventListener('click', () => send({cmd: 'step'}));
document.getElementById('btnReset').addEventListener('click', () => send({cmd: 'reset'}));

document.getElementById('modeSelect').addEventListener('change', (e) => {
  send({cmd: 'set_mode', mode: e.target.value});
  document.getElementById('goalSection').style.display =
    e.target.value === 'interactive' ? 'block' : 'none';
});

const fpsSlider = document.getElementById('fpsSlider');
const fpsVal = document.getElementById('fpsVal');
fpsSlider.addEventListener('input', () => {
  fpsVal.textContent = fpsSlider.value;
  send({cmd: 'set_speed', fps: parseInt(fpsSlider.value)});
});

// Train modal
document.getElementById('btnTrain').addEventListener('click', () => {
  document.getElementById('trainModal').classList.add('open');
  document.getElementById('trainProgress').style.width = '0%';
  document.getElementById('trainStatus').textContent = 'Ready';
});
document.getElementById('trainCancel').addEventListener('click', () => {
  document.getElementById('trainModal').classList.remove('open');
});
document.getElementById('trainStart').addEventListener('click', () => {
  const epochs = parseInt(document.getElementById('trainEpochs').value) || 150;
  send({cmd: 'train', epochs});
  document.getElementById('trainStatus').textContent = 'Starting...';
});

// --- Init ---

async function init() {
  // Check status
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (!data.has_model) {
      document.getElementById('noModelBadge').style.display = 'block';
    }
  } catch (e) {
    console.error('Status check failed', e);
  }

  // Load goals
  try {
    const res = await fetch('/api/goals');
    const data = await res.json();
    goals = data.goals || [];
    const goalList = document.getElementById('goalList');
    for (const g of goals) {
      const btn = document.createElement('button');
      btn.className = 'goal-btn';
      btn.textContent = g;
      btn.addEventListener('click', () => send({cmd: 'set_goal', goal: g}));
      goalList.appendChild(btn);
    }
  } catch (e) {
    console.error('Goals fetch failed', e);
  }

  connect();
}

init();
