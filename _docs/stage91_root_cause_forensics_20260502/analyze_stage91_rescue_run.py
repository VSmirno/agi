#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path


HOST = "cuda@192.168.98.56"
REMOTE_RUN = Path(
    "/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/"
    "hyper_stage91_rescue_robustness_20260502T113506Z"
)
LOCAL_ROOT = Path(
    "/home/yorick/agi-stage90r-world-model-guardrails/_docs/"
    "stage91_root_cause_forensics_20260502"
)
REPORT_JSON = LOCAL_ROOT / "analysis_report.json"
REPORT_MD = LOCAL_ROOT / "analysis_report.md"

FILES = {
    "bounded7": "raw/seed7_bounded_eval.json",
    "seed7": "raw/seed_7_eval.json",
    "seed17": "raw/seed_17_eval.json",
    "seed27": "raw/seed_27_eval.json",
    "seed37": "raw/seed_37_eval.json",
    "seed47": "raw/seed_47_eval.json",
    "summary": "summaries/multiseed_compare_summary.json",
    "bounded_cmd": "commands/bounded_compare_seed7.cmd",
    "seed7_cmd": "commands/eval_seed_7.cmd",
    "seed17_cmd": "commands/eval_seed_17.cmd",
    "seed27_cmd": "commands/eval_seed_27.cmd",
    "runner_log": "logs/runner.log",
    "bounded_log": "logs/seed7_bounded_eval.log",
    "seed7_log": "logs/seed_7_eval.log",
    "seed17_log": "logs/seed_17_eval.log",
    "seed27_log": "logs/seed_27_eval.log",
}


def ssh_cat(relative_path: str) -> str:
    remote_path = REMOTE_RUN / relative_path
    return subprocess.check_output(
        ["ssh", HOST, "cat", str(remote_path)],
        text=True,
    )


def ssh_json(relative_path: str) -> dict:
    return json.loads(ssh_cat(relative_path))


def episode_compact(ep: dict) -> dict:
    rescue_steps = [row["step"] for row in ep["rescue_trace"]]
    activation_reasons = Counter(
        row.get("activation_reason") or row.get("trigger") or "none"
        for row in ep["rescue_trace"]
    )
    override_sources = Counter(
        row.get("override_source") or "none" for row in ep["rescue_trace"]
    )
    excerpt_steps = [row["step"] for row in ep["local_trace_excerpt"]]
    excerpt_controllers = Counter(
        row.get("controller") or "unknown" for row in ep["local_trace_excerpt"]
    )
    return {
        "episode_id": ep["episode_id"],
        "steps": ep["episode_steps"],
        "death": ep["death_cause"],
        "rescues": ep["n_rescue_events"],
        "rescue_step_min": min(rescue_steps) if rescue_steps else None,
        "rescue_step_max": max(rescue_steps) if rescue_steps else None,
        "activation_reasons": dict(activation_reasons),
        "override_sources": dict(override_sources),
        "excerpt_steps": excerpt_steps,
        "excerpt_controllers": dict(excerpt_controllers),
    }


def eval_compact(data: dict) -> dict:
    summary = data["summary"]
    return {
        "config": data["config"],
        "summary": {
            key: summary[key]
            for key in [
                "avg_survival",
                "rescue_rate",
                "learner_control_fraction",
                "planner_dependence",
                "early_hostile_deaths_without_rescue",
                "hostile_deaths_without_rescue",
                "controller_distribution",
                "death_cause_breakdown",
                "rescue_activation_reason_distribution",
                "rescue_override_source_distribution",
                "rescue_trigger_distribution",
            ]
        },
        "episodes": [episode_compact(ep) for ep in data["episodes"]],
    }


def command_diff(a: str, b: str) -> list[str]:
    a_parts = a.strip().split()
    b_parts = b.strip().split()
    diffs = []
    max_len = max(len(a_parts), len(b_parts))
    for i in range(max_len):
        left = a_parts[i] if i < len(a_parts) else "<missing>"
        right = b_parts[i] if i < len(b_parts) else "<missing>"
        if left != right:
            diffs.append(f"arg[{i}]: {left} != {right}")
    return diffs


def build_report() -> dict:
    bounded = ssh_json(FILES["bounded7"])
    seed7 = ssh_json(FILES["seed7"])
    seed17 = ssh_json(FILES["seed17"])
    seed27 = ssh_json(FILES["seed27"])
    seed37 = ssh_json(FILES["seed37"])
    seed47 = ssh_json(FILES["seed47"])
    summary = ssh_json(FILES["summary"])

    per_seed_summary = summary["per_seed"]
    per_seed_summary_map = {
        str(entry["seed"]): entry for entry in per_seed_summary
    }
    raw_seed_avgs = {
        "7": seed7["summary"]["avg_survival"],
        "17": seed17["summary"]["avg_survival"],
        "27": seed27["summary"]["avg_survival"],
        "37": seed37["summary"]["avg_survival"],
        "47": seed47["summary"]["avg_survival"],
    }

    return {
        "remote_run_root": str(REMOTE_RUN),
        "inspected_paths": {name: str(REMOTE_RUN / rel) for name, rel in FILES.items()},
        "command_diffs": {
            "bounded7_vs_seed7": command_diff(
                ssh_cat(FILES["bounded_cmd"]), ssh_cat(FILES["seed7_cmd"])
            ),
            "seed7_vs_seed17": command_diff(
                ssh_cat(FILES["seed7_cmd"]), ssh_cat(FILES["seed17_cmd"])
            ),
            "seed7_vs_seed27": command_diff(
                ssh_cat(FILES["seed7_cmd"]), ssh_cat(FILES["seed27_cmd"])
            ),
        },
        "bounded7": eval_compact(bounded),
        "seed7": eval_compact(seed7),
        "seed17": eval_compact(seed17),
        "seed27": eval_compact(seed27),
        "seed37": eval_compact(seed37),
        "seed47": eval_compact(seed47),
        "summary_check": {
            "baseline_avg_survival": summary["baseline_avg_survival"],
            "candidate_mean_avg_survival": summary["candidate_mean_avg_survival"],
            "candidate_vs_baseline_delta": summary["candidate_vs_baseline_delta"],
            "status": summary["status"],
            "per_seed_summary": per_seed_summary_map,
            "raw_seed_avg_survival": raw_seed_avgs,
        },
        "log_snippets": {
            "runner_log": ssh_cat(FILES["runner_log"]).splitlines(),
            "bounded_log": ssh_cat(FILES["bounded_log"]).splitlines(),
            "seed7_log": ssh_cat(FILES["seed7_log"]).splitlines(),
            "seed17_log": ssh_cat(FILES["seed17_log"]).splitlines(),
            "seed27_log": ssh_cat(FILES["seed27_log"]).splitlines(),
        },
    }


def write_markdown(report: dict) -> None:
    def p_seed(seed_key: str) -> dict:
        return report[seed_key]["summary"]

    lines = [
        "# Stage91 Rescue Robustness Forensics",
        "",
        f"Remote run root: `{report['remote_run_root']}`",
        "",
        "## Command diff checks",
        f"- bounded7_vs_seed7: {report['command_diffs']['bounded7_vs_seed7'] or ['out path only']}",
        f"- seed7_vs_seed17: {report['command_diffs']['seed7_vs_seed17']}",
        f"- seed7_vs_seed27: {report['command_diffs']['seed7_vs_seed27']}",
        "",
        "## Seed summaries",
    ]
    for key in ["bounded7", "seed7", "seed17", "seed27", "seed37", "seed47"]:
        summary = report[key]["summary"]
        lines.append(
            "- {}: avg_survival={}, rescue_rate={}, learner_control_fraction={}, "
            "planner_dependence={}, controller_distribution={}, death_cause_breakdown={}".format(
                key,
                summary["avg_survival"],
                summary["rescue_rate"],
                summary["learner_control_fraction"],
                summary["planner_dependence"],
                summary["controller_distribution"],
                summary["death_cause_breakdown"],
            )
        )
    lines.extend(
        [
            "",
            "## Summary dataflow check",
            "- per_seed summary avg_survival: {}".format(
                {
                    seed: report["summary_check"]["per_seed_summary"][seed]["avg_survival"]
                    for seed in sorted(report["summary_check"]["per_seed_summary"])
                }
            ),
            "- raw seed avg_survival: {}".format(
                report["summary_check"]["raw_seed_avg_survival"]
            ),
            "",
            "## Episode compact views",
        ]
    )
    for key in ["bounded7", "seed7", "seed17", "seed27"]:
        lines.append(f"### {key}")
        for ep in report[key]["episodes"]:
            lines.append(
                "- ep{episode_id}: steps={steps}, death={death}, rescues={rescues}, "
                "rescue_step_min={rescue_step_min}, rescue_step_max={rescue_step_max}, "
                "activation_reasons={activation_reasons}, override_sources={override_sources}".format(
                    **ep
                )
            )
        lines.append("")
    REPORT_MD.write_text("\n".join(lines))


def main() -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True))
    write_markdown(report)
    print(REPORT_JSON)
    print(REPORT_MD)


if __name__ == "__main__":
    main()
