// Causal graph SVG renderer
// Renders ConceptStore causal links as a directed graph

const GRAPH_NODES = {
  // Fixed positions for known concepts (x, y in SVG viewBox 600x180)
  tree:       { x: 40,  y: 40 },
  wood:       { x: 130, y: 40 },
  table:      { x: 220, y: 40 },
  wood_pickaxe:  { x: 310, y: 30 },
  stone:      { x: 400, y: 30 },
  stone_item: { x: 400, y: 70 },
  stone_pickaxe: { x: 490, y: 30 },
  coal:       { x: 400, y: 110 },
  coal_item:  { x: 490, y: 110 },
  iron:       { x: 530, y: 70 },
  iron_item:  { x: 580, y: 70 },
  water:      { x: 40,  y: 110 },
  cow:        { x: 130, y: 110 },
  zombie:     { x: 40,  y: 155 },
  diamond:    { x: 580, y: 30 },
  empty:      { x: 220, y: 110 },
  furnace:    { x: 310, y: 110 },
};

// Parse confidence key: "tree_do_wood" → {from: "tree", action: "do", to: "wood"}
function parseKey(key) {
  const parts = key.split('_');
  if (parts.length < 3) return null;
  // Action is typically the second part
  // Handle multi-word concepts: try action at index 1, rest is result
  const from = parts[0];
  const action = parts[1];
  const to = parts.slice(2).join('_');
  return { from, action, to };
}

let lastConfidence = {};

function renderCausalGraph(confidence) {
  // Only re-render if changed
  const key = JSON.stringify(confidence);
  if (key === JSON.stringify(lastConfidence)) return;
  lastConfidence = confidence;

  const svg = document.getElementById('causalSvg');
  svg.innerHTML = '';

  const edges = [];
  const nodeSet = new Set();

  for (const [k, conf] of Object.entries(confidence)) {
    const parsed = parseKey(k);
    if (!parsed) continue;
    edges.push({ ...parsed, confidence: conf });
    nodeSet.add(parsed.from);
    nodeSet.add(parsed.to);
  }

  // Draw edges
  for (const edge of edges) {
    const fromPos = GRAPH_NODES[edge.from];
    const toPos = GRAPH_NODES[edge.to];
    if (!fromPos || !toPos) continue;

    const strokeWidth = 1 + edge.confidence * 3; // 1-4px
    const opacity = 0.3 + edge.confidence * 0.7;

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', fromPos.x);
    line.setAttribute('y1', fromPos.y);
    line.setAttribute('x2', toPos.x);
    line.setAttribute('y2', toPos.y);
    line.setAttribute('stroke', '#8c8caa');
    line.setAttribute('stroke-width', strokeWidth);
    line.setAttribute('stroke-opacity', opacity);
    line.setAttribute('marker-end', 'url(#arrow)');
    svg.appendChild(line);

    // Action label on edge midpoint
    const mx = (fromPos.x + toPos.x) / 2;
    const my = (fromPos.y + toPos.y) / 2 - 6;
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', mx);
    text.setAttribute('y', my);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '9');
    text.setAttribute('fill', '#666');
    text.textContent = edge.action;
    svg.appendChild(text);
  }

  // Draw nodes
  for (const name of nodeSet) {
    const pos = GRAPH_NODES[name];
    if (!pos) continue;

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', pos.x);
    circle.setAttribute('cy', pos.y);
    circle.setAttribute('r', 12);
    circle.setAttribute('fill', '#3498db');
    circle.setAttribute('fill-opacity', '0.8');
    svg.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pos.x);
    text.setAttribute('y', pos.y + 24);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '10');
    text.textContent = name;
    svg.appendChild(text);
  }

  // Arrow marker (defined once)
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  marker.setAttribute('id', 'arrow');
  marker.setAttribute('viewBox', '0 0 10 10');
  marker.setAttribute('refX', '22');
  marker.setAttribute('refY', '5');
  marker.setAttribute('markerWidth', '6');
  marker.setAttribute('markerHeight', '6');
  marker.setAttribute('orient', 'auto-start-reverse');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
  path.setAttribute('fill', '#8c8caa');
  marker.appendChild(path);
  defs.appendChild(marker);
  svg.insertBefore(defs, svg.firstChild);
}
