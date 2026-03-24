"""Генератор сводной страницы demo_output/index.html.

Показывает ссылки и краткие метрики всех трёх демо-отчётов.
Запускать ПОСЛЕ того как все три демо завершились.

Usage:
    python -m snks.experiments.exp_demo_index --output demo_output
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>СНКС Demo — Обзор результатов</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #080818;
  color: #d0d0f0;
  font-family: 'Segoe UI', 'Inter', system-ui, monospace;
  line-height: 1.7;
  min-height: 100vh;
}}
.header {{
  background: linear-gradient(135deg, #0d1f4a 0%, #1a0a3e 60%, #0d2a3a 100%);
  padding: 50px 60px 40px;
  border-bottom: 2px solid #1e1e5e;
}}
.header h1 {{
  font-size: 2.5em;
  font-weight: 900;
  background: linear-gradient(90deg, #4ECDC4, #45B7D1, #BB8FCE, #F7DC6F);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 10px;
}}
.header .sub {{
  color: #6677aa;
  font-size: 0.95em;
  margin-top: 12px;
  max-width: 780px;
}}
.tagline {{
  background: #0e0e22;
  padding: 18px 60px;
  border-bottom: 1px solid #181838;
  color: #7788bb;
  font-size: 0.88em;
}}
.content {{
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 60px;
}}
.section-title {{
  color: #5566aa;
  font-size: 0.78em;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 20px;
  padding-bottom: 8px;
  border-bottom: 1px solid #1a1a3e;
}}
.demos-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 24px;
  margin-bottom: 40px;
}}
.demo-card {{
  background: #0e0e22;
  border: 1px solid #222248;
  border-radius: 14px;
  padding: 28px;
  text-decoration: none;
  color: inherit;
  display: block;
  transition: border-color 0.2s, transform 0.2s;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}}
.demo-card:hover {{
  border-color: #4a4a8e;
  transform: translateY(-2px);
}}
.demo-card .icon {{
  font-size: 2.2em;
  margin-bottom: 12px;
}}
.demo-card h2 {{
  font-size: 1.2em;
  color: #99aadd;
  margin-bottom: 6px;
  font-weight: 700;
}}
.demo-card .desc {{
  color: #556688;
  font-size: 0.85em;
  margin-bottom: 18px;
  line-height: 1.5;
}}
.demo-card .metrics {{
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}}
.metric {{
  background: #111128;
  border: 1px solid #202050;
  border-radius: 8px;
  padding: 10px 14px;
  text-align: center;
  flex: 1;
  min-width: 80px;
}}
.metric .val {{
  font-size: 1.4em;
  font-weight: 700;
  background: linear-gradient(90deg, #4ECDC4, #45B7D1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}
.metric.pass .val {{
  background: linear-gradient(90deg, #4ECDC4, #96CEB4);
  -webkit-background-clip: text; background-clip: text;
}}
.metric.warn .val {{
  background: linear-gradient(90deg, #F7DC6F, #f0a830);
  -webkit-background-clip: text; background-clip: text;
}}
.metric .lbl {{
  font-size: 0.68em;
  color: #5566aa;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 3px;
}}
.badge {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.7em;
  font-weight: 700;
  letter-spacing: 0.05em;
  margin-left: 6px;
  vertical-align: middle;
}}
.badge-pass {{ background: #1a3a1a; color: #4ECDC4; border: 1px solid #2a5a2a; }}
.badge-fail {{ background: #3a1a1a; color: #FF6B6B; border: 1px solid #5a2a2a; }}
.badge-pend {{ background: #1a1a3a; color: #8899cc; border: 1px solid #2a2a5a; }}
.btn {{
  display: inline-block;
  margin-top: 18px;
  padding: 8px 20px;
  background: linear-gradient(135deg, #1a2a5e, #0d2244);
  border: 1px solid #2a3a7e;
  border-radius: 8px;
  color: #7799dd;
  font-size: 0.83em;
  font-weight: 600;
  text-decoration: none;
  letter-spacing: 0.03em;
}}
.btn:hover {{ background: linear-gradient(135deg, #2a3a7e, #1a2a5e); color: #99bbff; }}
.info-section {{
  background: #0e0e22;
  border: 1px solid #222248;
  border-radius: 14px;
  padding: 28px;
  margin-bottom: 28px;
}}
.info-section h2 {{
  color: #7788bb;
  font-size: 1.05em;
  margin-bottom: 16px;
  font-weight: 600;
}}
.key-fact {{
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 14px;
  color: #8899bb;
  font-size: 0.9em;
}}
.key-fact .num {{
  background: linear-gradient(135deg, #2a3a7e, #1a2a5e);
  border: 1px solid #3a4a9e;
  border-radius: 50%;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8em;
  font-weight: 700;
  color: #7799dd;
  flex-shrink: 0;
  margin-top: 2px;
}}
.footer {{
  text-align: center;
  padding: 24px;
  color: #33334a;
  font-size: 0.78em;
  border-top: 1px solid #141430;
  margin-top: 36px;
}}
</style>
</head>
<body>
<div class="header">
  <h1>СНКС: Демонстрация результатов</h1>
  <p style="color:#8899bb;font-size:1.0em">
    Система Непрерывного Когнитивного Синтеза — AGI-архитектура на базе FHN-осцилляторов
  </p>
  <div class="sub">
    Ни одна из демонстраций не использует метки при обучении.
    Нет backpropagation. Нет градиентного спуска.
    Только локальные правила STDP и динамика осцилляторов.
  </div>
</div>

<div class="tagline">
  <strong>Железо:</strong> AMD Radeon 8060S · 92 GB VRAM · ROCm 7.2 ·
  <strong>Стек:</strong> Python 3.11 · PyTorch 2.7+rocm7.2 ·
  <strong>Дата:</strong> {timestamp}
</div>

<div class="content">

<p class="section-title">Демонстрационные эксперименты</p>

<div class="demos-grid">

  <!-- Shapes -->
  <a class="demo-card" href="shapes_report.html">
    <div class="icon">⬟</div>
    <h2>Геометрические фигуры
      <span class="badge {shapes_badge}">{shapes_status}</span>
    </h2>
    <div class="desc">
      10 классов фигур: круг, квадрат, треугольник… — синтетические данные.
      Система самостоятельно формирует 10 кластеров без единой метки.
      Быстрый sanity-check: NMI &gt; 0.7 за ~20 мин.
    </div>
    <div class="metrics">
      <div class="metric {shapes_nmi_cls}">
        <div class="val">{shapes_nmi}</div>
        <div class="lbl">NMI</div>
      </div>
      <div class="metric">
        <div class="val">20K</div>
        <div class="lbl">Осцилляторов</div>
      </div>
      <div class="metric">
        <div class="val">800</div>
        <div class="lbl">Изображений</div>
      </div>
    </div>
    <span class="btn">Открыть отчёт →</span>
  </a>

  <!-- MNIST -->
  <a class="demo-card" href="mnist_report.html">
    <div class="icon">✎</div>
    <h2>Рукописные цифры MNIST
      <span class="badge {mnist_badge}">{mnist_status}</span>
    </h2>
    <div class="desc">
      10 цифр (0–9), 200 изображений на класс = 2000 images.
      Система учит 50K осцилляторов, формирует кластеры по цифрам.
      Визуализация: t-SNE, тепловая карта, примеры изображений.
    </div>
    <div class="metrics">
      <div class="metric {mnist_nmi_cls}">
        <div class="val">{mnist_nmi}</div>
        <div class="lbl">NMI</div>
      </div>
      <div class="metric">
        <div class="val">50K</div>
        <div class="lbl">Осцилляторов</div>
      </div>
      <div class="metric">
        <div class="val">2K</div>
        <div class="lbl">Изображений</div>
      </div>
    </div>
    <span class="btn">Открыть отчёт →</span>
  </a>

  <!-- Continual -->
  <a class="demo-card" href="continual_report.html">
    <div class="icon">♻</div>
    <h2>Непрерывное обучение
      <span class="badge {cont_badge}">{cont_status}</span>
    </h2>
    <div class="desc">
      Phase A: учим цифры 0–4 · Phase B: учим 5–9 (0–4 больше не показываются).
      Демонстрация: нет catastrophic forgetting.
      Retention = NMI(0–4 после B) / NMI(0–4 до B).
    </div>
    <div class="metrics">
      <div class="metric {cont_ret_cls}">
        <div class="val">{cont_retention}</div>
        <div class="lbl">Retention</div>
      </div>
      <div class="metric">
        <div class="val">30K</div>
        <div class="lbl">Осцилляторов</div>
      </div>
      <div class="metric">
        <div class="val">{cont_nmi}</div>
        <div class="lbl">NMI all 10</div>
      </div>
    </div>
    <span class="btn">Открыть отчёт →</span>
  </a>

</div>

<div class="info-section">
  <h2>Почему это работает — ключевые принципы СНКС</h2>
  <div class="key-fact">
    <div class="num">1</div>
    <div><strong style="color:#99bbdd">FHN-осцилляторы</strong> (FitzHugh-Nagumo) — математическая модель нейрона.
    При подаче стимула часть нейронов синхронизируется, формируя устойчивый паттерн (СКС — Синхронизированный Кластер Состояний).</div>
  </div>
  <div class="key-fact">
    <div class="num">2</div>
    <div><strong style="color:#99bbdd">STDP</strong> (Spike-Timing-Dependent Plasticity) — локальное обучение:
    если нейрон A срабатывает до нейрона B, связь A→B усиливается. Это единственный механизм обучения.
    Никаких глобальных градиентов.</div>
  </div>
  <div class="key-fact">
    <div class="num">3</div>
    <div><strong style="color:#99bbdd">Rate-based детекция</strong> — кластеры определяются по частоте спайков.
    Нейроны с частотой выше порога (mean + 3σ) считаются активными. Быстро: O(N).</div>
  </div>
  <div class="key-fact">
    <div class="num">4</div>
    <div><strong style="color:#99bbdd">NMI</strong> (Normalized Mutual Information) — метрика качества кластеризации.
    Показывает, насколько кластеры системы соответствуют истинным классам.
    NMI=1.0 = идеальное соответствие, NMI=0 = случайность.</div>
  </div>
</div>

</div>

<div class="footer">
  СНКС MVP v0.4.0 · Этапы 0–8 завершены · 17 экспериментов PASS ·
  FHN+STDP · No backpropagation · {timestamp}
</div>
</body>
</html>"""


def _read_metric(path: Path, key: str, default="—"):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get(key, default)
        return val
    except Exception:
        return default


def run(output_dir: str = "demo_output") -> None:
    out = Path(output_dir)

    # Shapes
    sm = out / "shapes_metrics.json"
    shapes_nmi_val = _read_metric(sm, "final_nmi", None)
    if shapes_nmi_val is not None:
        shapes_nmi = f"{shapes_nmi_val:.3f}"
        shapes_ok = shapes_nmi_val > 0.7
        shapes_badge = "badge-pass" if shapes_ok else "badge-fail"
        shapes_status = "PASS ✓" if shapes_ok else "FAIL"
        shapes_nmi_cls = "pass" if shapes_ok else "warn"
    else:
        shapes_nmi = "—"
        shapes_badge = "badge-pend"
        shapes_status = "В процессе..."
        shapes_nmi_cls = ""

    # MNIST
    mm = out / "mnist_metrics.json"
    mnist_nmi_val = _read_metric(mm, "final_nmi", None)
    if mnist_nmi_val is not None:
        mnist_nmi = f"{mnist_nmi_val:.3f}"
        mnist_ok = mnist_nmi_val > 0.6
        mnist_badge = "badge-pass" if mnist_ok else "badge-fail"
        mnist_status = "PASS ✓" if mnist_ok else "FAIL"
        mnist_nmi_cls = "pass" if mnist_ok else "warn"
    else:
        mnist_nmi = "—"
        mnist_badge = "badge-pend"
        mnist_status = "В процессе..."
        mnist_nmi_cls = ""

    # Continual
    cm = out / "continual_metrics.json"
    cont_ret_val = _read_metric(cm, "retention_pct", None)
    cont_nmi_val = _read_metric(cm, "nmi_all", None)
    if cont_ret_val is not None:
        cont_retention = f"{cont_ret_val:.0f}%"
        cont_ok = cont_ret_val >= 85
        cont_badge = "badge-pass" if cont_ok else "badge-fail"
        cont_status = "PASS ✓" if cont_ok else "FAIL"
        cont_ret_cls = "pass" if cont_ok else "warn"
        cont_nmi = f"{cont_nmi_val:.3f}" if cont_nmi_val is not None else "—"
    else:
        cont_retention = "—"
        cont_badge = "badge-pend"
        cont_status = "В процессе..."
        cont_ret_cls = ""
        cont_nmi = "—"

    html = _INDEX_HTML.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M"),
        shapes_nmi=shapes_nmi,
        shapes_badge=shapes_badge,
        shapes_status=shapes_status,
        shapes_nmi_cls=shapes_nmi_cls,
        mnist_nmi=mnist_nmi,
        mnist_badge=mnist_badge,
        mnist_status=mnist_status,
        mnist_nmi_cls=mnist_nmi_cls,
        cont_retention=cont_retention,
        cont_badge=cont_badge,
        cont_status=cont_status,
        cont_ret_cls=cont_ret_cls,
        cont_nmi=cont_nmi,
    )

    index_path = out / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(f"[INDEX] Сохранён: {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="demo_output")
    args = parser.parse_args()
    run(args.output)
