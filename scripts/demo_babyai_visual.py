"""Visual demo: BabyAI executor in action.

Generates an HTML page with animated step-by-step replay of the agent
executing BabyAI instructions. Each frame shows the grid, agent position,
and what the language pipeline decided.

Usage:
    python scripts/demo_babyai_visual.py
    # Opens demos/babyai_demo.html in browser
"""

from __future__ import annotations

import base64
import io
import os
import sys
import webbrowser

sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid
import numpy as np
from PIL import Image

from snks.language.babyai_executor import BabyAIExecutor
from snks.language.grid_perception import GridPerception
from snks.language.grid_navigator import GridNavigator
from snks.language.grounding_map import GroundingMap
from snks.language.chunker import RuleBasedChunker

ACTION_NAMES = {
    0: "⬅️ turn left", 1: "➡️ turn right", 2: "⬆️ forward",
    3: "✊ pickup", 4: "📦 drop", 5: "🚪 toggle", 6: "✅ done",
}


def frame_to_b64(frame: np.ndarray, scale: int = 6) -> str:
    """Convert RGB numpy frame to base64 PNG string, scaled up."""
    img = Image.fromarray(frame)
    w, h = img.size
    img = img.resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_episode(env_name: str, seed: int) -> dict | None:
    """Run one episode, capture frames and metadata."""
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    mission = obs["mission"]

    gmap = GroundingMap()
    perc = GridPerception(gmap)
    chunker = RuleBasedChunker()
    nav = GridNavigator()

    # Parse
    chunks = chunker.chunk(mission)
    action_text, object_text, attr_text = BabyAIExecutor._extract_roles(chunks)
    if not action_text:
        env.close()
        return None

    # Perceive
    uw = env.unwrapped
    perc.perceive(uw.grid, tuple(uw.agent_pos), int(uw.agent_dir))
    target_word = f"{attr_text} {object_text}" if attr_text else object_text
    target_obj = perc.find_object(target_word)
    if target_obj is None and attr_text:
        target_obj = perc.find_object(object_text)
    if target_obj is None:
        env.close()
        return None

    # Plan
    is_goto = action_text == "go to"
    nav_actions = nav.plan_path(
        uw.grid, tuple(uw.agent_pos), int(uw.agent_dir),
        target_obj.pos, stop_adjacent=True,
    )
    terminal = None
    if action_text == "pick up":
        terminal = 3
    elif action_text in ("open", "toggle"):
        terminal = 5

    all_actions = list(nav_actions)
    if terminal is not None:
        all_actions.append(terminal)

    # Capture initial frame
    frames = []
    frame0 = env.render()
    frames.append({
        "img": frame_to_b64(frame0),
        "action": "🎬 START",
        "pos": list(map(int, uw.agent_pos)),
        "dir": int(uw.agent_dir),
        "reward": 0,
    })

    # Execute step by step
    total_reward = 0
    for action in all_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame = env.render()
        frames.append({
            "img": frame_to_b64(frame),
            "action": ACTION_NAMES.get(action, str(action)),
            "pos": list(map(int, uw.agent_pos)),
            "dir": int(uw.agent_dir),
            "reward": round(float(total_reward), 3),
        })
        if terminated:
            break

    env.close()

    return {
        "mission": mission,
        "env": env_name,
        "seed": seed,
        "parsed": {
            "action": action_text,
            "attr": attr_text,
            "object": object_text,
        },
        "target": {
            "word": target_word,
            "type": target_obj.obj_type,
            "color": target_obj.color,
            "pos": list(target_obj.pos),
        },
        "plan": [ACTION_NAMES.get(a, str(a)) for a in all_actions],
        "frames": frames,
        "success": total_reward > 0,
        "total_steps": len(all_actions),
    }


def build_html(episodes: list[dict], out_path: str):
    """Generate self-contained HTML demo page."""
    import json

    episodes_json = json.dumps(episodes, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SNKS Stage 24c — BabyAI Demo</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    min-height: 100vh;
}}
.header {{
    text-align: center;
    padding: 30px 20px 10px;
    border-bottom: 1px solid #1a1a2e;
}}
.header h1 {{
    font-size: 28px;
    color: #00d4ff;
    letter-spacing: 2px;
    margin-bottom: 8px;
}}
.header .subtitle {{
    color: #666;
    font-size: 13px;
}}
.episodes {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 30px;
    padding: 30px;
}}
.episode {{
    background: #12121f;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    width: 520px;
    overflow: hidden;
    transition: border-color 0.3s;
}}
.episode:hover {{ border-color: #00d4ff44; }}
.episode.success {{ border-left: 3px solid #00ff88; }}
.episode.fail {{ border-left: 3px solid #ff4444; }}
.ep-header {{
    padding: 16px 20px;
    border-bottom: 1px solid #1e1e3a;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.ep-header .mission {{
    font-size: 15px;
    color: #ffcc00;
    font-style: italic;
}}
.ep-header .badge {{
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 99px;
    font-weight: bold;
}}
.badge.pass {{ background: #00ff8822; color: #00ff88; }}
.badge.fail {{ background: #ff444422; color: #ff4444; }}
.ep-parse {{
    padding: 10px 20px;
    background: #0d0d18;
    font-size: 12px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}
.ep-parse .tag {{
    background: #1a1a30;
    padding: 3px 10px;
    border-radius: 4px;
}}
.ep-parse .tag .label {{ color: #666; }}
.ep-parse .tag .val {{ color: #00d4ff; }}
.player {{
    padding: 16px 20px;
    text-align: center;
}}
.player img {{
    border: 2px solid #1e1e3a;
    border-radius: 8px;
    image-rendering: pixelated;
    max-width: 100%;
}}
.controls {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
}}
.controls button {{
    background: #1a1a30;
    color: #e0e0e0;
    border: 1px solid #2a2a4a;
    padding: 6px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-size: 13px;
    transition: all 0.2s;
}}
.controls button:hover {{ background: #2a2a4a; border-color: #00d4ff; }}
.controls button.playing {{ background: #00d4ff22; border-color: #00d4ff; color: #00d4ff; }}
.step-info {{
    margin-top: 10px;
    font-size: 13px;
    color: #888;
    min-height: 20px;
}}
.step-info .action-name {{ color: #ffcc00; }}
.step-info .reward {{ color: #00ff88; }}
.timeline {{
    display: flex;
    gap: 3px;
    justify-content: center;
    margin-top: 10px;
    flex-wrap: wrap;
}}
.timeline .dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #1e1e3a;
    cursor: pointer;
    transition: all 0.2s;
}}
.timeline .dot.active {{ background: #00d4ff; transform: scale(1.3); }}
.timeline .dot.done {{ background: #00d4ff44; }}
.stats {{
    text-align: center;
    padding: 20px;
    color: #666;
    font-size: 13px;
    border-top: 1px solid #1a1a2e;
}}
.stats span {{ color: #00d4ff; }}
.pipeline {{
    padding: 12px 20px 16px;
    text-align: center;
}}
.pipeline .arrow {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #444;
}}
.pipeline .arrow .node {{
    background: #1a1a30;
    border: 1px solid #2a2a4a;
    padding: 3px 8px;
    border-radius: 4px;
    color: #888;
    font-size: 11px;
}}
.pipeline .arrow .node.active {{
    border-color: #00d4ff;
    color: #00d4ff;
}}
</style>
</head>
<body>

<div class="header">
    <h1>SNKS — BabyAI Demo</h1>
    <div class="subtitle">
        Stage 24c: text instruction → parse → navigate → execute &nbsp;|&nbsp;
        Language pipeline closes the loop
    </div>
</div>

<div class="pipeline" style="padding-top:20px">
    <div class="arrow">
        <span class="node active">📝 Text</span> →
        <span class="node active">🔍 Chunker</span> →
        <span class="node active">🎯 Perception</span> →
        <span class="node active">🗺️ BFS Nav</span> →
        <span class="node active">🤖 Execute</span> →
        <span class="node active">✅ Verify</span>
    </div>
</div>

<div class="episodes" id="episodes"></div>

<div class="stats" id="stats"></div>

<script>
const episodes = {episodes_json};

const container = document.getElementById('episodes');
const statsEl = document.getElementById('stats');

let totalSuccess = 0;
let totalSteps = 0;

episodes.forEach((ep, idx) => {{
    if (ep.success) totalSuccess++;
    totalSteps += ep.total_steps;

    const div = document.createElement('div');
    div.className = 'episode ' + (ep.success ? 'success' : 'fail');
    div.innerHTML = `
        <div class="ep-header">
            <span class="mission">"${{ep.mission}}"</span>
            <span class="badge ${{ep.success ? 'pass' : 'fail'}}">
                ${{ep.success ? '✓ SUCCESS' : '✗ FAIL'}}
            </span>
        </div>
        <div class="ep-parse">
            <span class="tag"><span class="label">action:</span> <span class="val">${{ep.parsed.action}}</span></span>
            ${{ep.parsed.attr ? `<span class="tag"><span class="label">attr:</span> <span class="val">${{ep.parsed.attr}}</span></span>` : ''}}
            <span class="tag"><span class="label">object:</span> <span class="val">${{ep.parsed.object}}</span></span>
            <span class="tag"><span class="label">target:</span> <span class="val">${{ep.target.color}} ${{ep.target.type}} @ (${{ep.target.pos[0]}},${{ep.target.pos[1]}})</span></span>
        </div>
        <div class="player">
            <img id="frame-${{idx}}" src="data:image/png;base64,${{ep.frames[0].img}}" />
            <div class="controls">
                <button onclick="stepBack(${{idx}})">◀ Prev</button>
                <button id="play-${{idx}}" onclick="togglePlay(${{idx}})">▶ Play</button>
                <button onclick="stepForward(${{idx}})">Next ▶</button>
            </div>
            <div class="timeline" id="timeline-${{idx}}">
                ${{ep.frames.map((_, fi) =>
                    `<div class="dot ${{fi === 0 ? 'active' : ''}}" onclick="goToFrame(${{idx}}, ${{fi}})"></div>`
                ).join('')}}
            </div>
            <div class="step-info" id="info-${{idx}}">
                <span class="action-name">🎬 START</span>
            </div>
        </div>
    `;
    container.appendChild(div);
}});

statsEl.innerHTML = `
    <span>${{totalSuccess}}</span>/${{episodes.length}} episodes succeeded &nbsp;|&nbsp;
    avg <span>${{(totalSteps / episodes.length).toFixed(1)}}</span> steps &nbsp;|&nbsp;
    <span>${{episodes.length}}</span> episodes total
`;

// Player state
const state = episodes.map(() => ({{ frame: 0, playing: false, timer: null }}));

function updateFrame(idx) {{
    const ep = episodes[idx];
    const s = state[idx];
    const f = ep.frames[s.frame];
    document.getElementById(`frame-${{idx}}`).src = 'data:image/png;base64,' + f.img;

    const dots = document.querySelectorAll(`#timeline-${{idx}} .dot`);
    dots.forEach((d, i) => {{
        d.className = 'dot' + (i === s.frame ? ' active' : i < s.frame ? ' done' : '');
    }});

    let info = `<span class="action-name">${{f.action}}</span>`;
    info += ` &nbsp; step ${{s.frame}}/${{ep.frames.length - 1}}`;
    if (f.reward > 0) info += ` &nbsp; <span class="reward">reward: ${{f.reward}}</span>`;
    document.getElementById(`info-${{idx}}`).innerHTML = info;
}}

function stepForward(idx) {{
    const ep = episodes[idx];
    if (state[idx].frame < ep.frames.length - 1) {{
        state[idx].frame++;
        updateFrame(idx);
    }} else {{
        stopPlay(idx);
    }}
}}

function stepBack(idx) {{
    if (state[idx].frame > 0) {{
        state[idx].frame--;
        updateFrame(idx);
    }}
}}

function goToFrame(idx, fi) {{
    state[idx].frame = fi;
    updateFrame(idx);
}}

function togglePlay(idx) {{
    if (state[idx].playing) {{
        stopPlay(idx);
    }} else {{
        state[idx].playing = true;
        document.getElementById(`play-${{idx}}`).className = 'playing';
        document.getElementById(`play-${{idx}}`).textContent = '⏸ Pause';
        state[idx].timer = setInterval(() => stepForward(idx), 500);
    }}
}}

function stopPlay(idx) {{
    state[idx].playing = false;
    clearInterval(state[idx].timer);
    document.getElementById(`play-${{idx}}`).className = '';
    document.getElementById(`play-${{idx}}`).textContent = '▶ Play';
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("Generating BabyAI visual demo...")

    # Collect diverse episodes
    scenarios = [
        ("BabyAI-GoToObj-v0", 42),
        ("BabyAI-GoToObj-v0", 7),
        ("BabyAI-GoToObj-v0", 123),
        ("BabyAI-PickupLoc-v0", 10),
        ("BabyAI-PickupLoc-v0", 33),
        ("BabyAI-PickupLoc-v0", 55),
        ("BabyAI-GoToLocalS6N4-v0", 0),   # multiple objects
        ("BabyAI-GoToLocalS6N4-v0", 8),   # multiple objects
    ]

    episodes = []
    for env_name, seed in scenarios:
        print(f"  Running {env_name} seed={seed}...", end=" ")
        ep = run_episode(env_name, seed)
        if ep:
            episodes.append(ep)
            status = "SUCCESS" if ep["success"] else "FAIL"
            print(f'{ep["mission"]} → {status} ({ep["total_steps"]} steps)')
        else:
            print("skipped (parse failed)")

    out_path = "demos/babyai_demo.html"
    build_html(episodes, out_path)
    print(f"\nDone! Opening {out_path}...")

    # Open in browser
    abs_path = os.path.abspath(out_path)
    webbrowser.open(f"file:///{abs_path}")


if __name__ == "__main__":
    main()
