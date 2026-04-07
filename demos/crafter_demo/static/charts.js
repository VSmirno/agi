// Sparkline charts for episode metrics

const CHART_COLOR = '#3498db';
const CHART_BG = 'rgba(255,255,255,0.03)';

function drawSparkline(canvasId, data, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data || data.length === 0) return;

  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * (window.devicePixelRatio || 1);
  canvas.height = rect.height * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

  const w = rect.width;
  const h = rect.height;
  const pad = 2;

  // Clear
  ctx.fillStyle = CHART_BG;
  ctx.fillRect(0, 0, w, h);

  if (data.length < 2) {
    // Single point — just show value
    ctx.fillStyle = color || CHART_COLOR;
    ctx.font = '11px monospace';
    ctx.fillText(String(data[0]), pad, h / 2 + 4);
    return;
  }

  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;

  const stepX = (w - pad * 2) / (data.length - 1);

  ctx.beginPath();
  ctx.strokeStyle = color || CHART_COLOR;
  ctx.lineWidth = 1.5;

  for (let i = 0; i < data.length; i++) {
    const x = pad + i * stepX;
    const y = h - pad - ((data[i] - min) / range) * (h - pad * 2);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill area under line
  ctx.lineTo(pad + (data.length - 1) * stepX, h - pad);
  ctx.lineTo(pad, h - pad);
  ctx.closePath();
  ctx.fillStyle = (color || CHART_COLOR) + '20';
  ctx.fill();

  // Last value label
  const lastVal = data[data.length - 1];
  ctx.fillStyle = color || CHART_COLOR;
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(String(lastVal), w - pad, 12);
}

function updateSparklines(history) {
  if (history.lengths) drawSparkline('chartLength', history.lengths, '#3498db');
  if (history.resources) drawSparkline('chartResources', history.resources, '#2ecc71');
  if (history.encounters) drawSparkline('chartEncounters', history.encounters, '#e74c3c');
}
