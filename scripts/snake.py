"""Kahn1 plays Snake: one ChoiceQuestion per tick, read from the logits.

Every tick the board becomes the state and Kahn1 picks up / down / left /
right. Nothing is generated and nothing is parsed: the move is the argmax of
the four option letters, so an illegal answer cannot happen by construction.
All live games advance together in one batched engine call per tick.

Two ways of showing the board:

- grid:   the ASCII board and the rules, nothing else. The model has to find
          the head, the food and the obstacles itself.
- senses: the same board plus what lies one step in each direction (wall,
          body, empty) and whether that step brings the head closer to the
          food. The model still decides; it no longer has to do geometry.

Baselines give the scale: random (any move but straight back) and greedy (closest
safe move towards the food, a one-line heuristic).

Usage:
    python scripts/snake.py --model checkpoints/qwen_merged --games 20
    python scripts/snake.py --policies kahn1-senses greedy --replay reports/snake_replay.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sysone.types import ChoiceQuestion, Query

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

@dataclass
class Game:
    size: int
    rng: random.Random
    snake: list[tuple[int, int]] = field(default_factory=list)  # head first
    heading: str = "right"
    food: tuple[int, int] = (0, 0)
    score: int = 0
    steps: int = 0
    since_food: int = 0
    over: str = ""  # "" while alive, else the cause of death
    frames: list[dict] = field(default_factory=list)

    def __post_init__(self):
        c = self.size // 2
        self.snake = [(c, c), (c - 1, c), (c - 2, c)]
        self._place_food()
        self._record(None, None)

    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.snake]
        self.food = self.rng.choice(free) if free else (-1, -1)

    def _record(self, move, probs):
        self.frames.append({"snake": [list(p) for p in self.snake], "food": list(self.food),
                            "score": self.score, "move": move, "probs": probs})

    def cell(self, p: tuple[int, int]) -> str:
        x, y = p
        if not (0 <= x < self.size and 0 <= y < self.size):
            return "wall"
        # The tail moves away this tick unless the snake eats, so it is safe.
        if p in self.snake[:-1]:
            return "body"
        return "food" if p == self.food else "empty"

    def step(self, move: str, probs: dict | None = None, max_hunger: int = 0):
        dx, dy = MOVES[move]
        hx, hy = self.snake[0]
        nxt = (hx + dx, hy + dy)
        what = self.cell(nxt)
        self.steps += 1
        if what in ("wall", "body"):
            self.over = f"hit {what}"
            self._record(move, probs)
            return
        self.snake.insert(0, nxt)
        self.heading = move
        if nxt == self.food:
            self.score += 1
            self.since_food = 0
            self._place_food()
        else:
            self.snake.pop()
            self.since_food += 1
        if max_hunger and self.since_food >= max_hunger:
            self.over = "starved"  # looping forever is not playing
        self._record(move, probs)

    def render(self) -> str:
        rows = ["#" * (self.size + 2)]
        body = set(self.snake[1:])
        for y in range(self.size):
            line = "#"
            for x in range(self.size):
                p = (x, y)
                line += ("H" if p == self.snake[0] else "o" if p in body
                         else "F" if p == self.food else ".")
            rows.append(line + "#")
        rows.append(rows[0])
        return "\n".join(rows)


def dist(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# States shown to Kahn1
# ---------------------------------------------------------------------------

RULES = (
    "You control the snake in a game of Snake on a {n}x{n} board.\n"
    "Legend: H = snake head, o = snake body, F = food, . = empty, # = wall.\n"
    "Row 1 is the top. Moving into a wall or into the body ends the game.\n"
    "Eating the food makes the snake longer and scores a point."
)


def state_grid(g: Game) -> str:
    hx, hy = g.snake[0]
    fx, fy = g.food
    return (
        RULES.format(n=g.size) + "\n\n" + g.render() + "\n\n"
        f"Score: {g.score}. Snake length: {len(g.snake)}. "
        f"Last move: {g.heading}."
    )


def state_senses(g: Game) -> str:
    head = g.snake[0]
    now = dist(head, g.food)
    lines = []
    for m, (dx, dy) in MOVES.items():
        nxt = (head[0] + dx, head[1] + dy)
        what = g.cell(nxt)
        if what in ("wall", "body"):
            lines.append(f"- {m}: {what}, moving there ends the game")
        elif what == "food":
            lines.append(f"- {m}: the food, moving there scores a point")
        else:
            d = dist(nxt, g.food)
            trend = "closer to" if d < now else "farther from"
            lines.append(f"- {m}: empty, {trend} the food ({d} steps away)")
    return (
        state_grid(g) + "\n\n"
        f"The food is {now} steps away. One step in each direction:\n" + "\n".join(lines)
    )


QUESTION = ChoiceQuestion(
    key="move",
    prompt="Which move should the snake make now to stay alive and reach the food?",
    options=list(MOVES),
    allow_other=False,
)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def greedy_move(g: Game, rng: random.Random) -> str:
    head = g.snake[0]
    safe = [m for m, (dx, dy) in MOVES.items()
            if g.cell((head[0] + dx, head[1] + dy)) not in ("wall", "body")]
    if not safe:
        return g.heading
    return min(safe, key=lambda m: (dist((head[0] + MOVES[m][0], head[1] + MOVES[m][1]),
                                          g.food), rng.random()))


def play(policy: str, engine, games: int, size: int, seed: int, max_steps: int,
         max_hunger: int, permutations: int) -> list[Game]:
    # Same seeds for every policy: identical food sequences, a fair comparison.
    pool = [Game(size, random.Random(seed + i)) for i in range(games)]
    rng = random.Random(seed)
    while True:
        live = [g for g in pool if not g.over and g.steps < max_steps]
        if not live:
            return pool
        if policy.startswith("kahn1"):
            view = state_senses if policy == "kahn1-senses" else state_grid
            resp = engine.evaluate_batch(
                [Query(state=view(g), questions=[QUESTION]) for g in live],
                n_permutations=permutations)
            for g, r in zip(live, resp):
                a = r.answers["move"]
                g.step(a.choice, {k: round(v, 3) for k, v in a.probabilities.items()},
                       max_hunger)
        else:
            for g in live:
                if policy == "random":  # never straight back into its own neck
                    m = rng.choice([m for m in MOVES if m != OPPOSITE[g.heading]])
                else:
                    m = greedy_move(g, rng)
                g.step(m, None, max_hunger)


def summarize(policy: str, pool: list[Game], elapsed: float, decisions: int) -> dict:
    scores = [g.score for g in pool]
    causes: dict[str, int] = {}
    for g in pool:
        c = g.over or "step limit"
        causes[c] = causes.get(c, 0) + 1
    return {
        "policy": policy,
        "games": len(pool),
        "mean_score": round(statistics.mean(scores), 2),
        "max_score": max(scores),
        "mean_steps": round(statistics.mean(g.steps for g in pool), 1),
        "endings": causes,
        "ms_per_decision": round(1000 * elapsed / max(decisions, 1), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Okura66/Kahn1-Qwen2.5-3B")
    ap.add_argument("--policies", nargs="+",
                    default=["random", "greedy", "kahn1-grid", "kahn1-senses"],
                    choices=["random", "greedy", "kahn1-grid", "kahn1-senses"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--max-hunger", type=int, default=120,
                    help="steps without eating before a game counts as starved (0 = off)")
    ap.add_argument("--permutations", type=int, default=3,
                    help="option-order permutations averaged per decision (debiasing)")
    ap.add_argument("--replay", default="", help="write every game's frames to this JSON file")
    args = ap.parse_args()

    engine = None
    if any(p.startswith("kahn1") for p in args.policies):
        from sysone.engine import Engine, EngineConfig
        engine = Engine(EngineConfig(model=args.model, gpu_memory_utilization=0.80))
        engine._ensure_loaded()

    summaries, replays = [], {}
    for policy in args.policies:
        t0 = time.perf_counter()
        pool = play(policy, engine, args.games, args.size, args.seed, args.max_steps,
                    args.max_hunger, args.permutations)
        elapsed = time.perf_counter() - t0
        s = summarize(policy, pool, elapsed, sum(g.steps for g in pool))
        summaries.append(s)
        replays[policy] = [{"seed": args.seed + i, "score": g.score, "steps": g.steps,
                            "ending": g.over or "step limit", "frames": g.frames}
                           for i, g in enumerate(pool)]
        print(f"[{policy}] done in {elapsed:.1f}s", file=sys.stderr)

    print(f"\n{args.games} games per policy, {args.size}x{args.size} board, seed {args.seed}\n")
    print(f"{'policy':14s} {'mean':>6s} {'max':>4s} {'steps':>6s} {'ms/move':>8s}  endings")
    for s in summaries:
        endings = ", ".join(f"{k} {v}" for k, v in sorted(s["endings"].items()))
        print(f"{s['policy']:14s} {s['mean_score']:6.2f} {s['max_score']:4d} "
              f"{s['mean_steps']:6.1f} {s['ms_per_decision']:8.1f}  {endings}")

    if args.replay:
        out = Path(args.replay)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"size": args.size, "summaries": summaries,
                                   "games": replays}), encoding="utf-8")
        print(f"\nreplay written to {out}")


if __name__ == "__main__":
    main()
