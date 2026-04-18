# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/snks/`, organized by subsystem: `daf/`, `dcam/`, `agent/`, `language/`, `learning/`, `env/`, `pipeline/`, and `viz/`. Keep new modules inside the closest existing package instead of creating parallel top-level directories. Tests live in `tests/` with smaller suites under folders such as `tests/agent/`, `tests/learning/`, and `tests/metacog/`. Research notes, stage specs, and reports are under `docs/`; runnable demos and small web assets are in `demos/`; experiment and runtime YAML lives in `configs/`.

## Research Model & Architecture Rules
This repository follows the ideology in `docs/IDEOLOGY.md`: keep **facts**, **mechanisms**, **experience**, and **stimuli** separate. Stable world knowledge belongs in structured stores such as `configs/crafter_textbook.yaml`; generic logic belongs in Python mechanisms; episode-local observations belong in runtime state; motivation belongs in stimulus/evaluation layers. Do not patch missing reasoning with Crafter-specific `if` chains inside planners or policies. If behavior is wrong, first ask whether the bug is in textbook facts, mechanism design, runtime experience flow, or stimulus scoring.

## Build, Test, and Development Commands
Create an isolated environment and install the package in editable mode:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
Run the full suite with `pytest tests/ -x -q`. Run a focused regression with `pytest tests/test_stage59_lockedroom.py -v` or `pytest tests/learning/test_rule_nursery.py -v`. For local package imports, prefer `python -m pytest` if your shell environment is inconsistent.

## Coding Style & Naming Conventions
Use Python 3.11+ with 4-space indentation and PEP 8–style naming: `snake_case` for functions and modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants. Match the existing code style in neighboring files; this repository does not expose a mandatory formatter or linter config in `pyproject.toml`, so keep changes minimal and readable. Name new experiment files with the established pattern `expNN_description.py`.

## Stage-Driven Development
This project is roadmap-driven, not feature-driven. Before adding logic, check `docs/ROADMAP.md` and `docs/ASSUMPTIONS.md` for the current stage’s debt, gates, and accepted shortcuts. Prefer removing ideological debt over adding tactical eval hacks. Apply `docs/STAGE_REVIEW_CRITERIA.md` before declaring a strategy, stage, or result successful, `docs/ANTI_TUNING_CHECKLIST.md` before treating a gain as architectural rather than environment-specific, and `docs/CONCEPT_SUCCESS_CRITERIA.md` before claiming that the overall architecture or concept is proven. If a change introduces a new simplification, limitation, or known failure mode, record it in `docs/ASSUMPTIONS.md`; if it changes stage behavior materially, update the matching design/report/spec document in `docs/`.

## Testing Guidelines
Tests use `pytest` with `-v --tb=short` defaults from `pyproject.toml`. Add or update tests alongside behavioral changes, especially for new stage logic, agent policies, and learning components. Follow the current naming pattern `tests/test_<feature>.py` and keep fixtures in `tests/conftest.py` when they are shared across suites.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit style with scopes, for example `fix(post_mortem): ...`, `feat(exp136): ...`, and `docs(stage88): ...`. Keep commits atomic and scoped to one change. Pull requests should include a short problem statement, the approach taken, and exact verification commands run. Attach screenshots or demo notes when changing `demos/`, visualization output, or other user-facing artifacts.

## Documentation & Research Notes
When behavior changes, update the relevant stage/spec/report under `docs/` in the same branch. Repository-level plans and design writeups already exist in `docs/design/` and `docs/superpowers/`; extend them instead of creating ad hoc note files. Treat `docs/IDEOLOGY.md` as the thinking model, `docs/ROADMAP.md` as the sequencing source, `docs/ASSUMPTIONS.md` as the log of temporary constraints and unresolved gaps, `docs/STAGE_REVIEW_CRITERIA.md` as the required stage evaluation checklist, `docs/ANTI_TUNING_CHECKLIST.md` as the guardrail against environment-specific self-deception, and `docs/CONCEPT_SUCCESS_CRITERIA.md` as the higher bar for concept-level claims.
