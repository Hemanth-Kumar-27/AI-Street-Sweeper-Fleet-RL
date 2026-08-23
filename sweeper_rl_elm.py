"""
Reinforcement Learning with Extreme Learning Machine (RL-ELM / Q-ELM) for
Coverage Path Planning of Autonomous Sweeper Robots.

This module implements the core methodology described in the capstone proposal:
  1. A Gymnasium-style grid-world coverage environment (CoverageGridEnv)
  2. An Extreme Learning Machine used as a fast, closed-form-trained Q-value
     function approximator (ELM)
  3. A Q-learning training loop that refits the ELM's output layer after every
     episode instead of back-propagating through a deep network (train_q_elm)
  4. A greedy rollout of the learned policy, with stuck/oscillation detection
     (rollout_greedy)
  5. A classical boustrophedon (zigzag) baseline planner for comparison
     (boustrophedon_baseline)
  6. A nearest-neighbour fallback sweep that mops up any cells the learned
     policy missed, guaranteeing 100% coverage (fallback_complete)

This is a Python/numpy port of the identical algorithm used in the
accompanying interactive HTML demo (Sweeper_RL_ELM_Path_Planner.html), so
results/behaviour should match. Use this module to run offline experiments,
sweep hyperparameters, and generate report/slide figures.

Dependencies: numpy (required), matplotlib (only for the demo plot at the
bottom of this file).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Grid cell codes
OUTSIDE = 0
DIRTY = 1
OBSTACLE = 2

# 4-connected moves: North, South, West, East
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# --------------------------------------------------------------------------
# Extreme Learning Machine
# --------------------------------------------------------------------------
class ELM:
    """Single-hidden-layer network with a random, fixed hidden layer and a
    closed-form (ridge-regression) output layer. Used here as a fast
    Q-value function approximator: unlike a back-propagation-trained
    network, `fit` re-solves the output weights in one linear-algebra step,
    which is what lets the agent retrain almost instantly as new experience
    arrives (the "Q-ELM" update rule)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # random, FIXED hidden layer (never trained) + bias column
        self.W_in = rng.uniform(-1.0, 1.0, size=(input_dim + 1, hidden_dim))
        self.W_out = np.zeros((hidden_dim, output_dim))

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # append bias input
        return 1.0 / (1.0 + np.exp(-(Xb @ self.W_in)))  # sigmoid activation

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._hidden(X) @ self.W_out

    def fit(self, X: np.ndarray, Y: np.ndarray, lam: float = 1e-2) -> None:
        """Ridge-regression closed-form solve: W_out = (HtH + lam*I)^-1 Ht Y"""
        H = self._hidden(np.asarray(X))
        Y = np.asarray(Y)
        HtH = H.T @ H + lam * np.eye(H.shape[1])
        HtY = H.T @ Y
        self.W_out = np.linalg.solve(HtH, HtY)


# --------------------------------------------------------------------------
# Coverage path planning environment
# --------------------------------------------------------------------------
class CoverageGridEnv:
    """Gymnasium-style environment: an agent must visit every DIRTY cell of
    a grid at least once while avoiding OBSTACLE/OUTSIDE cells.

    State: a (2*win_r+1)^2 local window around the agent (−1 obstacle/outside,
    0.3 already covered, 1 not yet covered) plus 4 scalar features (remaining
    coverage fraction, step-budget fraction, and the normalized direction to
    the nearest uncovered cell — a mild shaping signal that keeps training
    fast enough to run interactively).
    Actions: 0=N, 1=S, 2=W, 3=E.
    Reward: +1.2 newly covering a cell, −0.6 revisiting, −1.0 invalid move,
    −0.03 step cost, +5.0 bonus on full coverage.
    """

    def __init__(self, mask: np.ndarray, win_r: int = 2, max_steps: Optional[int] = None):
        self.mask = mask
        self.rows, self.cols = mask.shape
        self.win_r = win_r
        self.dirty_count = int(np.sum(mask == DIRTY))
        self.max_steps = max_steps or (self.dirty_count * 3 + 20)
        self.observation_dim = (2 * win_r + 1) ** 2 + 4
        self.action_dim = len(ACTIONS)
        self.covered = None
        self.pos = None
        self.covered_count = 0
        self.steps = 0
        self.done = False
        self.uncovered = set()
        self.reset()

    def _walkable(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols and self.mask[r, c] == DIRTY

    def _find_start(self, preferred=None):
        if preferred is not None and self._walkable(*preferred):
            return preferred
        ys, xs = np.where(self.mask == DIRTY)
        return (int(ys[0]), int(xs[0])) if len(ys) else None

    def reset(self, start: Optional[tuple] = None):
        self.covered = np.zeros_like(self.mask, dtype=bool)
        self.pos = self._find_start(start) or (0, 0)
        self.covered[self.pos] = True
        self.covered_count = 1
        self.steps = 0
        self.done = False
        self.uncovered = set(zip(*np.where(self.mask == DIRTY)))
        self.uncovered.discard(self.pos)
        return self._get_obs()

    def _nearest_uncovered_dir(self):
        if not self.uncovered:
            return 0.0, 0.0
        pr, pc = self.pos
        best = min(self.uncovered, key=lambda p: abs(p[0] - pr) + abs(p[1] - pc))
        dr, dc = best[0] - pr, best[1] - pc
        norm = max(1, abs(dr) + abs(dc))
        return dr / norm, dc / norm

    def _get_obs(self) -> np.ndarray:
        feats = []
        pr, pc = self.pos
        for dr in range(-self.win_r, self.win_r + 1):
            for dc in range(-self.win_r, self.win_r + 1):
                r, c = pr + dr, pc + dc
                if not (0 <= r < self.rows and 0 <= c < self.cols) or self.mask[r, c] != DIRTY:
                    feats.append(-1.0)
                elif self.covered[r, c]:
                    feats.append(0.3)
                else:
                    feats.append(1.0)
        remaining_frac = 1.0 - self.covered_count / self.dirty_count
        step_frac = self.steps / self.max_steps
        gdr, gdc = self._nearest_uncovered_dir()
        feats.extend([remaining_frac, step_frac, gdr, gdc])
        return np.array(feats, dtype=np.float64)

    def step(self, action: int):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1
        if not self._walkable(nr, nc):
            reward = -1.0
        else:
            self.pos = (nr, nc)
            if not self.covered[nr, nc]:
                self.covered[nr, nc] = True
                self.covered_count += 1
                reward = 1.2
                self.uncovered.discard((nr, nc))
            else:
                reward = -0.6
            reward -= 0.03
        fully_covered = self.covered_count >= self.dirty_count
        self.done = fully_covered or self.steps >= self.max_steps
        if fully_covered:
            reward += 5.0
        return self._get_obs(), reward, self.done, fully_covered, {}


# --------------------------------------------------------------------------
# Q-ELM training (Q-learning with an ELM as the Q-function approximator)
# --------------------------------------------------------------------------
@dataclass
class TrainResult:
    elm: ELM
    env: CoverageGridEnv
    best_coverage: float
    episodes_run: int
    history: list = field(default_factory=list)


def train_q_elm(
    mask: np.ndarray,
    episodes: int = 70,
    hidden_dim: int = 80,
    win_r: int = 2,
    gamma: float = 0.95,
    lam: float = 1.0,
    buffer_cap: int = 3000,
    seed: int = 42,
    start: Optional[tuple] = None,
    early_stop_streak: int = 6,
    min_episodes: int = 15,
    verbose: bool = False,
) -> TrainResult:
    env = CoverageGridEnv(mask, win_r=win_r)
    elm = ELM(env.observation_dim, hidden_dim, env.action_dim, seed=seed)
    init_state = env.reset(start)
    elm.fit(init_state[None, :], np.zeros((1, env.action_dim)), lam=1.0)

    buffer_X: list = []
    buffer_Y: list = []
    epsilon, eps_min = 0.9, 0.05
    eps_decay = (eps_min / 0.9) ** (1.0 / max(1, episodes - 1))
    rng = np.random.default_rng(seed)
    best_coverage = 0.0
    solved_streak = 0
    history = []
    episodes_run = 0

    for ep in range(episodes):
        state = env.reset(start)
        transitions = []
        done = False
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_dim))
            else:
                action = int(np.argmax(elm.predict(state[None, :])[0]))
            next_state, reward, done, fully_covered, _ = env.step(action)
            transitions.append((state, action, reward, next_state, done))
            state = next_state

        coverage = env.covered_count / env.dirty_count
        best_coverage = max(best_coverage, coverage)

        for s, a, r, s2, d in transitions:
            q_cur = elm.predict(s[None, :])[0].copy()
            target = r
            if not d:
                target += gamma * np.max(elm.predict(s2[None, :])[0])
            q_cur[a] = target
            buffer_X.append(s)
            buffer_Y.append(q_cur)
        if len(buffer_X) > buffer_cap:
            buffer_X = buffer_X[-buffer_cap:]
            buffer_Y = buffer_Y[-buffer_cap:]
        elm.fit(np.array(buffer_X), np.array(buffer_Y), lam=lam)

        epsilon = max(eps_min, epsilon * eps_decay)
        episodes_run = ep + 1

        # Track convergence via a *greedy* eval rollout (training coverage is
        # noisy because of epsilon-greedy exploration).
        eval_coverage = coverage
        if ep >= 8:
            eval_res = rollout_greedy(mask, elm, win_r=win_r, start=start)
            eval_coverage = eval_res["coverage"]
        solved_streak = solved_streak + 1 if eval_coverage >= 0.97 else 0

        history.append({"episode": ep, "coverage": coverage, "eval_coverage": eval_coverage, "epsilon": epsilon})
        if verbose and (ep % 10 == 0 or ep == episodes - 1):
            print(f"  episode {ep + 1}/{episodes}  coverage={coverage:.2f}  eval={eval_coverage:.2f}  epsilon={epsilon:.2f}")

        if solved_streak >= early_stop_streak and ep >= min(min_episodes, episodes - 1):
            if verbose:
                print(f"  converged early at episode {ep + 1} (eval coverage >= 97% for {early_stop_streak} episodes)")
            break

    return TrainResult(elm=elm, env=env, best_coverage=best_coverage, episodes_run=episodes_run, history=history)


def rollout_greedy(
    mask: np.ndarray,
    elm: ELM,
    win_r: int = 2,
    start: Optional[tuple] = None,
    max_steps: Optional[int] = None,
    stuck_limit: Optional[int] = None,
):
    """Run the learned policy greedily (argmax Q at every step). Includes
    stuck/oscillation detection: if the agent goes stuck_limit steps without
    covering a new cell, it stops early rather than looping forever — the
    caller's fallback_complete() mops up whatever's left."""
    env = CoverageGridEnv(mask, win_r=win_r, max_steps=max_steps)
    state = env.reset(start)
    path = [env.pos]
    stuck_limit = stuck_limit or max(30, win_r * 20)
    last_progress = env.covered_count
    stuck_for = 0
    done = False
    guard, max_guard = 0, env.max_steps + 10

    while not done and guard < max_guard:
        q = elm.predict(state[None, :])[0]
        order = np.argsort(-q)
        action = int(order[0])
        for a in order:
            dr, dc = ACTIONS[a]
            if env._walkable(env.pos[0] + dr, env.pos[1] + dc):
                action = int(a)
                break
        state, _reward, done, _fully_covered, _ = env.step(action)
        path.append(env.pos)
        guard += 1

        if env.covered_count > last_progress:
            last_progress = env.covered_count
            stuck_for = 0
        else:
            stuck_for += 1
            if stuck_for > stuck_limit:
                break

    return {
        "path": path,
        "covered_count": env.covered_count,
        "coverage": env.covered_count / env.dirty_count,
        "covered": env.covered,
    }


# --------------------------------------------------------------------------
# Classical baseline: row-major boustrophedon (zigzag) with segment linking
# --------------------------------------------------------------------------
def boustrophedon_baseline(mask: np.ndarray, start: Optional[tuple] = None) -> dict:
    """Guarantees full coverage of every walkable cell. Handles rows with
    multiple disconnected segments (e.g. around obstacles) by transiting to
    the nearest next segment."""
    rows, cols = mask.shape

    def walkable(r, c):
        return 0 <= r < rows and 0 <= c < cols and mask[r, c] == DIRTY

    def find_start(pref):
        if pref is not None and walkable(*pref):
            return pref
        ys, xs = np.where(mask == DIRTY)
        return (int(ys[0]), int(xs[0])) if len(ys) else None

    s = find_start(start)
    if s is None:
        return {"path": [], "covered_count": 0, "sweep_steps": 0, "transit_steps": 0, "total_steps": 0}

    path = [s]
    cur = s
    sweep_steps = 0
    transit_steps = 0

    rows_segments = []
    for r in range(rows):
        segs = []
        c = 0
        while c < cols:
            if mask[r, c] == DIRTY:
                c2 = c
                while c2 < cols and mask[r, c2] == DIRTY:
                    c2 += 1
                segs.append((c, c2 - 1))
                c = c2
            else:
                c += 1
        if segs:
            rows_segments.append((r, segs))

    start_idx = next(i for i, (r, _segs) in enumerate(rows_segments) if r == s[0])
    seq = rows_segments[start_idx:] + rows_segments[:start_idx][::-1]

    def move_to(target):
        nonlocal cur, transit_steps
        r, c = cur
        tr, tc = target
        while r != tr:
            r += 1 if r < tr else -1
            path.append((r, c))
            transit_steps += 1
        while c != tc:
            c += 1 if c < tc else -1
            path.append((r, c))
            transit_steps += 1
        cur = (r, c)

    direction = 1
    for row_idx, segs in seq:
        ordered_segs = segs if direction == 1 else [(b, a) for (a, b) in reversed(segs)]
        for c0, c1 in ordered_segs:
            entry = (row_idx, c0)
            if cur != entry:
                move_to(entry)
            c = c0
            step = 1 if c1 >= c0 else -1
            while c != c1:
                c += step
                path.append((row_idx, c))
                sweep_steps += 1
            cur = (row_idx, c1)
        direction *= -1

    covered_set = set(path)
    covered_count = sum(1 for r in range(rows) for c in range(cols) if mask[r, c] == DIRTY and (r, c) in covered_set)

    return {
        "path": path,
        "covered_count": covered_count,
        "sweep_steps": sweep_steps,
        "transit_steps": transit_steps,
        "total_steps": len(path) - 1,
    }


# --------------------------------------------------------------------------
# Nearest-neighbour fallback completion
# --------------------------------------------------------------------------
def fallback_complete(mask: np.ndarray, covered: np.ndarray, current_pos: tuple) -> dict:
    """Mops up any cells the learned policy missed, guaranteeing 100% final
    coverage regardless of how well training converged.

    Real road-network grids are frequently NOT one single 4-connected blob:
    a coarse cell grid, dead-end streets, or two ways that only touch
    diagonally in cell-space can split the walkable mask into several
    "islands" under strict N/S/E/W movement, even though the underlying
    street network is geometrically continuous. The original version gave
    up on any target it couldn't reach by a walkable-only BFS path
    (`remaining.discard(target)`), silently leaving those cells uncovered —
    on a fragmented grid that could mean the vast majority of cells never
    get swept, while also wasting time re-running a failed BFS for every
    single one of them individually.

    Fix: when no walkable path exists, bridge the gap with a short direct
    transit (same idea as the classical baseline's row-to-row transit) —
    a real vehicle can drive a short unmapped/disconnected link between two
    street segments. This guarantees the loop always converges to 100%
    coverage and terminates in a bounded number of steps.
    """
    rows, cols = mask.shape

    def walkable(r, c):
        return 0 <= r < rows and 0 <= c < cols and mask[r, c] == DIRTY

    covered = covered.copy()
    cur = current_pos
    extra_path = [cur]
    remaining = {(r, c) for r in range(rows) for c in range(cols) if mask[r, c] == DIRTY and not covered[r, c]}

    def bfs_path(frm, to):
        visited = {frm}
        prev = {}
        q = deque([frm])
        while q:
            cur2 = q.popleft()
            if cur2 == to:
                break
            for dr, dc in ACTIONS:
                nr, nc = cur2[0] + dr, cur2[1] + dc
                if walkable(nr, nc) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    prev[(nr, nc)] = cur2
                    q.append((nr, nc))
        if to not in visited:
            return None
        path = []
        node = to
        while node != frm:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def transit_to(frm, to):
        """Direct row-then-column hop, ignoring walkability — only used
        when bfs_path finds no walkable connection at all."""
        r, c = frm
        tr, tc = to
        path = []
        while r != tr:
            r += 1 if r < tr else -1
            path.append((r, c))
        while c != tc:
            c += 1 if c < tc else -1
            path.append((r, c))
        return path

    guard, max_guard = 0, len(remaining) * 2 + rows * cols + 10
    while remaining and guard < max_guard:
        guard += 1
        target = min(remaining, key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
        seg = bfs_path(cur, target) or transit_to(cur, target)
        for p in seg:
            extra_path.append(p)
            if walkable(*p):
                covered[p] = True
        cur = target
        covered[target] = True
        remaining = {p for p in remaining if not covered[p]}

    return {"extra_path": extra_path, "covered": covered}


# --------------------------------------------------------------------------
# Test-grid helper
# --------------------------------------------------------------------------
def make_room_with_obstacles(rows: int = 20, cols: int = 20, obstacle_boxes: Optional[Sequence[tuple]] = None) -> np.ndarray:
    """obstacle_boxes: list of (r0, c0, r1, c1) half-open rectangles."""
    mask = np.full((rows, cols), DIRTY, dtype=int)
    for (r0, c0, r1, c1) in (obstacle_boxes or []):
        mask[r0:r1, c0:c1] = OBSTACLE
    return mask


# --------------------------------------------------------------------------
# Demo: baseline vs RL-ELM, with a comparison plot
# --------------------------------------------------------------------------
if __name__ == "__main__":
    rows, cols = 20, 20
    obstacle_boxes = [(3, 3, 6, 6), (3, 10, 6, 13), (10, 3, 13, 6), (10, 10, 13, 13), (15, 15, 18, 18)]
    mask = make_room_with_obstacles(rows, cols, obstacle_boxes)
    dirty_total = int(np.sum(mask == DIRTY))
    print(f"Grid: {rows}x{cols}, {dirty_total} dirty cells, {len(obstacle_boxes)} obstacles\n")

    print("--- Classical baseline (boustrophedon) ---")
    t0 = time.time()
    baseline = boustrophedon_baseline(mask)
    baseline_ms = (time.time() - t0) * 1000
    print(f"steps={baseline['total_steps']}  coverage={baseline['covered_count']}/{dirty_total}  time={baseline_ms:.1f} ms\n")

    print("--- Training Q-ELM agent ---")
    t0 = time.time()
    result = train_q_elm(mask, episodes=70, hidden_dim=90, verbose=True)
    train_ms = (time.time() - t0) * 1000

    rollout = rollout_greedy(mask, result.elm)
    covered = rollout["covered"]
    extra_len = 0
    if rollout["coverage"] < 0.999:
        fb = fallback_complete(mask, rollout["covered"], rollout["path"][-1])
        covered = fb["covered"]
        extra_len = len(fb["extra_path"]) - 1
    total_steps = len(rollout["path"]) - 1 + extra_len
    total_ms = (time.time() - t0) * 1000
    final_coverage = float(np.sum(covered & (mask == DIRTY))) / dirty_total

    print(f"\nsteps={total_steps}  learned_coverage={rollout['coverage']*100:.0f}%  "
          f"final_coverage={final_coverage*100:.0f}%  train_time={train_ms:.0f} ms  total_time={total_ms:.0f} ms")

    base_redundancy = max(0.0, (baseline["total_steps"] - dirty_total) / dirty_total * 100)
    rl_redundancy = max(0.0, (total_steps - dirty_total) / dirty_total * 100)
    base_coverage_str = f"{baseline['covered_count']}/{dirty_total}"
    rl_coverage_str = f"{final_coverage * 100:.0f}%"
    base_redundancy_str = f"{base_redundancy:.0f}%"
    rl_redundancy_str = f"{rl_redundancy:.0f}%"

    print("\n--- Comparison ---")
    print(f"{'Metric':<28}{'Baseline':<15}{'RL-ELM':<15}")
    print(f"{'Total steps':<28}{baseline['total_steps']:<15}{total_steps:<15}")
    print(f"{'Coverage':<28}{base_coverage_str:<15}{rl_coverage_str:<15}")
    print(f"{'Redundant revisits':<28}{base_redundancy_str:<15}{rl_redundancy_str:<15}")
    print(f"{'Compute time (ms)':<28}{baseline_ms:<15.1f}{train_ms:<15.1f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        panels = [
            (axes[0], baseline["path"], f"Classical baseline\n{baseline['total_steps']} steps", "#ff9f43"),
            (axes[1], rollout["path"], f"RL-ELM (learned + fallback)\n{total_steps} steps", "#23c98f"),
        ]
        for ax, path, title, color in panels:
            ax.imshow(mask, cmap="gray_r", origin="upper", vmin=0, vmax=2)
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ax.plot(xs, ys, color=color, linewidth=1.2)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig("sweep_comparison.png", dpi=150)
        print("\nSaved comparison plot to sweep_comparison.png")
    except ImportError:
        print("\n(matplotlib not installed — skipping plot; pip install matplotlib to enable it)")
