"""Stage 76 Phase 4: Continuous decision loop.

Replaces Stage 75 phase6_survival plan-execution state machine. Every
step is a fresh decision from memory:

  1. Perceive → update spatial_map + tracker
  2. Encode raw state → SDR
  3. Query SDM for similar past episodes
  4. If enough similar episodes → deficit-weighted softmax over recalled actions
  5. Else → ConceptStore bootstrap plan (textbook instincts)
  6. Execute one step → write (state, action, next_state, body_delta) to SDM
  7. No plan state carried between steps (no plan_step_idx, no commitment)

Ideology compliance:
- No hardcoded drive list — uses tracker.observed_variables() / observed_max
- No derived features — state_sdr is raw bucket encoding
- No "higher is better" assumption — sign emerges from deficit × delta
- No if-else reflexes — action branch is SDM-scoring + softmax, fallback only
  if SDM is still cold
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_spatial_map import (
    CrafterSpatialMap,
    MOVE_ACTIONS,
    _step_toward,
)
from snks.agent.perception import (
    HomeostaticTracker,
    VisualField,
    outcome_to_verify,
    select_goal,
    verify_outcome,
)
from snks.memory.attention import AttentionWeights
from snks.memory.episodic_sdm import (
    Episode,
    EpisodicSDM,
    score_actions,
    select_action,
)
from snks.memory.state_encoder import StateEncoder


def perceive_tile_field(pixels: np.ndarray, segmenter: Any) -> VisualField:
    """Re-implementation of Stage 75 _perceive_segmenter.

    Runs the tile segmenter on the raw pixel frame and returns a VisualField
    populated with per-tile detections. The player's own tile is excluded
    (labeled "empty" in training data), and the 4 cardinal adjacent tiles
    determine `near_concept`.
    """
    from snks.agent.decode_head import NEAR_CLASSES

    px_tensor = torch.from_numpy(pixels) if isinstance(pixels, np.ndarray) else pixels
    class_ids, confidences = segmenter.classify_tiles(px_tensor)
    H, W = class_ids.shape
    cr, cc = H // 2, W // 2
    adjacent_pos = {(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)}

    vf = VisualField()
    for tr in range(H):
        for tc in range(W):
            cls_idx = int(class_ids[tr, tc].item())
            conf = float(confidences[tr, tc].item())
            if cls_idx < len(NEAR_CLASSES):
                cls_name = NEAR_CLASSES[cls_idx]
            else:
                cls_name = f"class_{cls_idx}"
            if cls_name == "empty":
                continue
            vf.detections.append((cls_name, conf, tr, tc))
            if (tr, tc) in adjacent_pos and conf > vf.near_similarity:
                vf.near_concept = cls_name
                vf.near_similarity = conf
    return vf


def _bootstrap_action(
    inventory: dict[str, int],
    tracker: HomeostaticTracker,
    vf: VisualField,
    spatial_map: CrafterSpatialMap,
    store: ConceptStore,
    player_pos: tuple[int, int],
    rng: np.random.RandomState,
    center_r: int,
    center_c: int,
) -> str:
    """Fallback strategy when SDM has too few similar episodes.

    Uses ConceptStore.plan() to derive an action from textbook instincts.
    Returns the first step's action, composed for make/place if needed.
    If no plan exists, picks a random move action for data gathering.
    """
    _goal, plan = select_goal(
        inventory,
        store,
        tracker=tracker,
        visual_field=vf,
        spatial_map=spatial_map,
    )
    if not plan:
        return str(rng.choice(MOVE_ACTIONS))

    step_plan = plan[0]
    target = step_plan.target

    # Self-actions (sleep, etc.)
    if target == "_self":
        action = step_plan.action
        if action.startswith("babble_"):
            action = action.replace("babble_", "")
        return action

    # Already adjacent to target
    if vf.near_concept == target:
        if step_plan.action == "do":
            return "do"
        if step_plan.action in ("make", "place"):
            return f"{step_plan.action}_{step_plan.expected_gain}"
        return step_plan.action

    # Navigate toward target via spatial map
    tgt_pos = spatial_map.find_nearest(target, player_pos)
    if tgt_pos is not None:
        return _step_toward(player_pos, tgt_pos, rng)

    # Fallback: head toward any visible tile of target
    for cid, _conf, gy, gx in vf.detections:
        if cid != target:
            continue
        dx = gx - center_c
        dy = gy - (center_r - 1)
        moves: list[str] = []
        if dx > 0:
            moves.append("move_right")
        elif dx < 0:
            moves.append("move_left")
        if dy > 0:
            moves.append("move_down")
        elif dy < 0:
            moves.append("move_up")
        if moves:
            return moves[rng.randint(len(moves))]

    return str(rng.choice(MOVE_ACTIONS))


def run_continuous_episode(
    env: Any,
    segmenter: Any,
    encoder: StateEncoder,
    sdm: EpisodicSDM,
    store: ConceptStore,
    tracker: HomeostaticTracker,
    rng: np.random.RandomState,
    max_steps: int = 500,
    temperature: float = 1.0,
    bootstrap_k: int = 5,
    similarity_threshold: float = 0.5,
    min_sdm_size: int = 500,
    attention: AttentionWeights | None = None,
    verbose: bool = False,
) -> dict:
    """Run a single continuous-learning episode.

    Every step: perceive → encode → recall → (SDM-action OR bootstrap) → step
    → write episode.

    Args:
        env: CrafterPixelEnv-compatible environment.
        segmenter: tile segmenter (Stage 75 checkpoint) with classify_tiles().
        encoder: StateEncoder instance (shared across episodes).
        sdm: EpisodicSDM instance (shared across episodes — persistent memory).
        store: ConceptStore (bootstrap source, also updated via verify_outcome).
        tracker: HomeostaticTracker (shared — observes rates + observed_max).
        rng: RandomState for fallback action picks + softmax sampling.
        max_steps: episode length cap.
        temperature: softmax temperature for action selection.
        bootstrap_k: minimum similar episodes needed to switch from bootstrap
            to SDM path. Below this threshold, ConceptStore is used.
        similarity_threshold: popcount-overlap ratio vs query popcount for
            an episode to count as "similar" during the bootstrap gate.
        min_sdm_size: minimum TOTAL episodes in SDM before the SDM path can
            trigger. Prevents cold-start where early similar-state matches
            yield uninformative scores.
        attention: optional AttentionWeights module. When provided, the
            per-step SDM recall uses a deficit-weighted mask built from
            learned per-variable bit relevance, and the module is updated
            after each step from (state_sdr, body_delta) correlations.
        verbose: if True, print per-step diagnostics.

    Returns:
        dict with episode metrics:
          length, sdm_size, final_inv, bootstrap_ratio, action_entropy,
          action_counts, cause_of_death.
    """
    from snks.encoder.tile_head_trainer import VIEWPORT_ROWS, VIEWPORT_COLS
    center_r = VIEWPORT_ROWS // 2
    center_c = VIEWPORT_COLS // 2

    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    prev_inv: dict[str, int] | None = None
    steps_taken = 0
    n_bootstrap = 0
    n_sdm = 0
    action_counts: Counter = Counter()
    cause_of_death = "alive"
    inv_after: dict[str, int] = dict(info.get("inventory", {}))

    for step in range(max_steps):
        steps_taken = step + 1
        inv = dict(info.get("inventory", {}))
        player_pos = tuple(info.get("player_pos", (32, 32)))
        px_player, py_player = int(player_pos[0]), int(player_pos[1])

        # Perception
        vf = perceive_tile_field(pixels, segmenter)
        spatial_map.update((px_player, py_player), vf.near_concept)
        for cid, _conf, gy, gx in vf.detections:
            wx = px_player + (gx - center_c)
            wy = py_player + (gy - (center_r - 1))
            spatial_map.update((wx, wy), cid)

        # Track body rates + observed_max
        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # Encode raw state → SDR
        state_sdr = encoder.encode(
            inventory=inv,
            visible_field=vf,
            spatial_map=spatial_map,
            player_pos=(px_player, py_player),
            body_variables=tracker.observed_variables() or None,
        )

        # Build a deficit-weighted attention mask from learned per-variable
        # bit relevance. Deficits are computed only over BODY variables
        # (health, food, drink, energy — whichever the tracker recognises
        # as having innate decay). Inventory items like wood and sapling
        # are excluded: being without them is not lethal, and their
        # observed_max grows unboundedly during collection, which would
        # swamp the mask with inventory-related bits at eval time.
        query_mask = None
        if attention is not None:
            body_vars = tracker.body_variables()
            deficits = {
                var: max(
                    0.0,
                    float(tracker.observed_max.get(var, 0) - inv.get(var, 0)),
                )
                for var in body_vars
            }
            query_mask = attention.query_mask(deficits)

        # Query memory — weighted by attention mask if available
        recalled = sdm.recall(state_sdr, top_k=20, mask=query_mask)
        # Bootstrap gate is binary popcount (not affected by attention)
        n_similar = sdm.count_similar(state_sdr, threshold_ratio=similarity_threshold)

        # Decide: SDM path only when (a) buffer has enough TOTAL diversity
        # and (b) enough of it is similar to the current query.
        sdm_ready = len(sdm) >= min_sdm_size
        if sdm_ready and n_similar >= bootstrap_k and recalled:
            action_scores = score_actions(recalled, inv, tracker)
            if action_scores:
                action_str = select_action(
                    action_scores, temperature=temperature, rng=rng,
                )
                n_sdm += 1
            else:
                action_str = _bootstrap_action(
                    inv, tracker, vf, spatial_map, store,
                    (px_player, py_player), rng, center_r, center_c,
                )
                n_bootstrap += 1
        else:
            action_str = _bootstrap_action(
                inv, tracker, vf, spatial_map, store,
                (px_player, py_player), rng, center_r, center_c,
            )
            n_bootstrap += 1

        action_counts[action_str] += 1

        if verbose:
            inv_summary = (
                f"H{inv.get('health',9)}F{inv.get('food',9)}"
                f"D{inv.get('drink',9)}E{inv.get('energy',9)}"
                f" W{inv.get('wood',0)}"
            )
            source = "SDM" if n_sdm and action_counts[action_str] == 1 else "BS"
            print(
                f"s{step:3d} {inv_summary} near={vf.near_concept:9s}"
                f" sdm={len(sdm)}/sim={n_similar} → [{source}] {action_str}"
            )

        # Execute
        inv_before = inv
        pixels, _reward, done, info = env.step(action_str)
        inv_after = dict(info.get("inventory", {}))

        # Compute body delta — only over tracked variables (no hardcoded list)
        observed_vars = tracker.observed_variables()
        if not observed_vars:
            # First step: nothing tracked yet; include everything observed so far
            observed_vars = set(inv.keys()) | set(inv_after.keys())
        body_delta = {
            var: int(inv_after.get(var, inv.get(var, 0)) - inv.get(var, 0))
            for var in observed_vars
        }

        # Update attention weights: bits active in state_sdr get credit or
        # blame for the observed body_delta. Over many episodes, this builds
        # a per-variable relevance map over SDR bits.
        if attention is not None:
            attention.update(state_sdr, body_delta)

        # Encode the next state for episodic storage
        next_vf = perceive_tile_field(pixels, segmenter)
        next_player_pos = tuple(info.get("player_pos", player_pos))
        next_state_sdr = encoder.encode(
            inventory=inv_after,
            visible_field=next_vf,
            spatial_map=spatial_map,
            player_pos=(int(next_player_pos[0]), int(next_player_pos[1])),
            body_variables=tracker.observed_variables() or None,
        )

        # Write episode — every step, unconditionally
        sdm.write(Episode(
            state_sdr=state_sdr,
            action=action_str,
            next_state_sdr=next_state_sdr,
            body_delta=body_delta,
            step=step,
        ))

        # Keep ConceptStore confidence up to date for bootstrap quality
        outcome = outcome_to_verify(action_str, inv_before, inv_after)
        if outcome:
            verify_outcome(vf.near_concept, action_str, outcome, store)

        prev_inv = inv

        if done:
            # Post-hoc telemetry: find any tracked variable that hit zero.
            # No hardcoded "health is death" assumption — whichever variable
            # the tracker observed that bottomed out is the cause.
            zeroed_vars = [
                var for var in tracker.observed_variables()
                if inv_after.get(var, 1) <= 0
            ]
            cause_of_death = ",".join(zeroed_vars) if zeroed_vars else "other"
            break

    # Action entropy over episode (information-theoretic exploration measure)
    total = sum(action_counts.values())
    if total > 0:
        probs = np.array([c / total for c in action_counts.values()])
        action_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        action_entropy = 0.0

    bootstrap_ratio = n_bootstrap / max(1, steps_taken)

    return {
        "length": steps_taken,
        "sdm_size": len(sdm),
        "final_inv": inv_after,
        "bootstrap_ratio": bootstrap_ratio,
        "sdm_ratio": n_sdm / max(1, steps_taken),
        "action_entropy": action_entropy,
        "action_counts": dict(action_counts),
        "cause_of_death": cause_of_death,
    }
