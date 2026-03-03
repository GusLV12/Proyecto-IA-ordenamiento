#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import time
import os
from dataclasses import dataclass
from collections import deque
import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# =========================
# CONFIG
# =========================

GOAL_STATE: Tuple[int, ...] = (
    1, 2, 3,
    4, 5, 6,
    7, 8, 0
)

MOVES: Dict[str, int] = {
    "UP": -3,
    "DOWN": 3,
    "LEFT": -1,
    "RIGHT": 1
}

# =========================
# DATA MODELS
# =========================

@dataclass(frozen=True)
class SolveResult:
    algo: str
    solved: bool
    time_s: float
    nodes_expanded: int
    solution_len: int
    solution_moves: Tuple[str, ...]


# =========================
# PUZZLE UTILITIES
# =========================

def print_board(state: Sequence[int]) -> None:
    for r in range(3):
        row = state[r * 3:(r + 1) * 3]
        print(" ".join(" " if x == 0 else str(x) for x in row))


def clear_console() -> None:
    # Cross-platform-ish clear
    os.system("cls" if os.name == "nt" else "clear")


def get_possible_moves(zero_index: int) -> List[str]:
    row = zero_index // 3
    col = zero_index % 3

    possible: List[str] = []
    if row > 0:
        possible.append("UP")
    if row < 2:
        possible.append("DOWN")
    if col > 0:
        possible.append("LEFT")
    if col < 2:
        possible.append("RIGHT")
    return possible


def apply_move(state: Tuple[int, ...], move: str) -> Tuple[int, ...]:
    zero_index = state.index(0)
    target_index = zero_index + MOVES[move]
    new_state = list(state)
    new_state[zero_index], new_state[target_index] = new_state[target_index], new_state[zero_index]
    return tuple(new_state)


def manhattan_distance(state: Tuple[int, ...]) -> int:
    # Precompute goal positions (micro-opt)
    # For 9 tiles, this is fine to compute once.
    goal_pos = {value: idx for idx, value in enumerate(GOAL_STATE)}

    dist = 0
    for i, value in enumerate(state):
        if value == 0:
            continue
        gi = goal_pos[value]
        x1, y1 = divmod(i, 3)
        x2, y2 = divmod(gi, 3)
        dist += abs(x1 - x2) + abs(y1 - y2)
    return dist


# =========================
# SOLVABILITY (3x3)
# =========================

def inversion_count(state: Sequence[int]) -> int:
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv


def is_solvable_3x3(state: Sequence[int]) -> bool:
    # For odd-width boards (3x3), solvable iff inversion count is even.
    return inversion_count(state) % 2 == 0


# =========================
# SEARCH ALGORITHMS
# =========================

def bfs(start_state: Tuple[int, ...]) -> SolveResult:
    t0 = time.perf_counter()

    if not is_solvable_3x3(start_state):
        return SolveResult("BFS", False, time.perf_counter() - t0, 0, 0, ())

    q = deque([(start_state, ())])  # (state, path_moves)
    visited = set()
    nodes = 0

    while q:
        state, path = q.popleft()

        if state in visited:
            continue

        visited.add(state)
        nodes += 1

        if state == GOAL_STATE:
            return SolveResult("BFS", True, time.perf_counter() - t0, nodes, len(path), path)

        z = state.index(0)
        for mv in get_possible_moves(z):
            nxt = apply_move(state, mv)
            if nxt not in visited:
                q.append((nxt, path + (mv,)))

    return SolveResult("BFS", False, time.perf_counter() - t0, nodes, 0, ())


def a_star(start_state: Tuple[int, ...]) -> SolveResult:
    t0 = time.perf_counter()

    if not is_solvable_3x3(start_state):
        return SolveResult("A*", False, time.perf_counter() - t0, 0, 0, ())

    # (f, g, state, path)
    heap: List[Tuple[int, int, Tuple[int, ...], Tuple[str, ...]]] = []
    g0 = 0
    h0 = manhattan_distance(start_state)
    heapq.heappush(heap, (g0 + h0, g0, start_state, ()))

    visited = set()
    nodes = 0

    while heap:
        f, g, state, path = heapq.heappop(heap)

        if state in visited:
            continue

        visited.add(state)
        nodes += 1

        if state == GOAL_STATE:
            return SolveResult("A*", True, time.perf_counter() - t0, nodes, len(path), path)

        z = state.index(0)
        for mv in get_possible_moves(z):
            nxt = apply_move(state, mv)
            if nxt in visited:
                continue
            ng = g + 1
            nf = ng + manhattan_distance(nxt)
            heapq.heappush(heap, (nf, ng, nxt, path + (mv,)))

    return SolveResult("A*", False, time.perf_counter() - t0, nodes, 0, ())


# =========================
# INSTANCE GENERATION
# =========================

def scramble_from_goal(steps: int, seed: Optional[int] = None) -> Tuple[int, ...]:
    """
    Genera un estado SOLVABLE aplicando 'steps' movimientos aleatorios desde el estado meta.
    Esto evita generar estados no alcanzables.
    """
    rng = random.Random(seed)
    state = GOAL_STATE
    last_move: Optional[str] = None

    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

    for _ in range(steps):
        z = state.index(0)
        moves = get_possible_moves(z)

        # Evitar deshacer el movimiento anterior (reduce bucles)
        if last_move and opposite[last_move] in moves and len(moves) > 1:
            moves.remove(opposite[last_move])

        mv = rng.choice(moves)
        state = apply_move(state, mv)
        last_move = mv

    return state


# =========================
# BENCHMARK + TABLE
# =========================

def run_benchmark(instances: List[Tuple[int, ...]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for idx, s in enumerate(instances, start=1):
        solv = is_solvable_3x3(s)

        r_bfs = bfs(s)
        r_ast = a_star(s)

        rows.append({
            "inst": idx,
            "solvable": solv,
            "bfs_time_s": r_bfs.time_s,
            "bfs_nodes": r_bfs.nodes_expanded,
            "bfs_len": r_bfs.solution_len,
            "astar_time_s": r_ast.time_s,
            "astar_nodes": r_ast.nodes_expanded,
            "astar_len": r_ast.solution_len,
        })

    return rows


def print_table(rows: List[Dict[str, object]]) -> None:
    # Try pandas for nice formatting; fallback to manual
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        # Redondeo para lectura
        if "bfs_time_s" in df:
            df["bfs_time_s"] = df["bfs_time_s"].map(lambda x: round(float(x), 6))
            df["astar_time_s"] = df["astar_time_s"].map(lambda x: round(float(x), 6))
        print(df.to_string(index=False))
        return
    except Exception:
        pass

    headers = [
        "inst", "solvable",
        "bfs_time_s", "bfs_nodes", "bfs_len",
        "astar_time_s", "astar_nodes", "astar_len"
    ]

    # Compute widths
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    sep = " | "
    line = "-+-".join("-" * widths[h] for h in headers)
    print(sep.join(h.ljust(widths[h]) for h in headers))
    print(line)
    for row in rows:
        print(sep.join(str(row[h]).ljust(widths[h]) for h in headers))


# =========================
# PLOTS
# =========================

def plot_results(rows: List[Dict[str, object]], out_prefix: str = "benchmark") -> None:
    import matplotlib.pyplot as plt  # type: ignore

    x = [int(r["inst"]) for r in rows]

    bfs_time = [float(r["bfs_time_s"]) for r in rows]
    ast_time = [float(r["astar_time_s"]) for r in rows]

    bfs_nodes = [int(r["bfs_nodes"]) for r in rows]
    ast_nodes = [int(r["astar_nodes"]) for r in rows]

    bfs_len = [int(r["bfs_len"]) for r in rows]
    ast_len = [int(r["astar_len"]) for r in rows]

    # 1) Tiempo
    plt.figure()
    plt.plot(x, bfs_time, marker="o", label="BFS")
    plt.plot(x, ast_time, marker="o", label="A* (Manhattan)")
    plt.title("Tiempo de ejecución por instancia")
    plt.xlabel("Instancia")
    plt.ylabel("Tiempo (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tiempo.png", dpi=160)

    # 2) Nodos expandidos
    plt.figure()
    plt.plot(x, bfs_nodes, marker="o", label="BFS")
    plt.plot(x, ast_nodes, marker="o", label="A* (Manhattan)")
    plt.title("Nodos expandidos por instancia")
    plt.xlabel("Instancia")
    plt.ylabel("Nodos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_nodos.png", dpi=160)

    # 3) Longitud de solución
    plt.figure()
    plt.plot(x, bfs_len, marker="o", label="BFS")
    plt.plot(x, ast_len, marker="o", label="A* (Manhattan)")
    plt.title("Longitud de la solución por instancia")
    plt.xlabel("Instancia")
    plt.ylabel("Movimientos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_longitud.png", dpi=160)

    print(f"\nGráficas guardadas como:\n"
          f"- {out_prefix}_tiempo.png\n"
          f"- {out_prefix}_nodos.png\n"
          f"- {out_prefix}_longitud.png")


# =========================
# ANIMATION (CONSOLE)
# =========================

def animate_solution(start: Tuple[int, ...], moves: Sequence[str], delay_s: float = 0.25) -> None:
    state = start
    clear_console()
    print("Estado inicial:")
    print_board(state)
    print("\nIniciando animación...\n")
    time.sleep(max(0.05, delay_s))

    for i, mv in enumerate(moves, start=1):
        state = apply_move(state, mv)
        clear_console()
        print(f"Paso {i}/{len(moves)}  |  Movimiento: {mv}\n")
        print_board(state)
        time.sleep(max(0.05, delay_s))

    print("\n✅ Animación finalizada.")


# =========================
# MAIN
# =========================

def parse_state_arg(s: str) -> Tuple[int, ...]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 9:
        raise argparse.ArgumentTypeError("El estado debe tener 9 números separados por coma (incluye 0).")
    if sorted(parts) != list(range(9)):
        raise argparse.ArgumentTypeError("El estado debe contener exactamente los números 0..8 sin repetir.")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="8-Puzzle: BFS vs A* con tabla, gráficas, solvencia y animación."
    )
    parser.add_argument("--instances", type=int, default=10, help="Cantidad de instancias a evaluar.")
    parser.add_argument("--scramble", type=int, default=20, help="Movimientos aleatorios desde meta para generar cada instancia.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla base para reproducibilidad.")
    parser.add_argument("--plot", action="store_true", help="Genera gráficas PNG de rendimiento.")
    parser.add_argument("--animate", action="store_true", help="Anima la solución (usa A* por defecto).")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay en segundos para la animación.")
    parser.add_argument("--state", type=parse_state_arg, default=None,
                        help="Estado manual: '1,2,3,4,0,6,7,5,8' (si lo pones, se evalúa solo esa instancia).")
    args = parser.parse_args()

    # Generación de instancias
    if args.state is not None:
        instances = [args.state]
    else:
        instances = [
            scramble_from_goal(args.scramble, seed=args.seed + i)
            for i in range(args.instances)
        ]

    # Mostrar solvencia rápida
    print("=== Instancias ===")
    for i, s in enumerate(instances, start=1):
        print(f"\nInstancia {i}: solvable={is_solvable_3x3(s)}  inv={inversion_count(s)}")
        print_board(s)

    # Benchmark + tabla
    print("\n=== Benchmark (BFS vs A*) ===")
    rows = run_benchmark(instances)
    print_table(rows)

    # Plots
    if args.plot:
        plot_results(rows, out_prefix="benchmark")

    # Animación (A*)
    if args.animate:
        start = instances[0]
        if not is_solvable_3x3(start):
            print("\n❌ No se puede animar: el estado NO es soluble.")
            return
        res = a_star(start)
        if not res.solved:
            print("\n❌ No se encontró solución para animar.")
            return
        animate_solution(start, res.solution_moves, delay_s=args.delay)


if __name__ == "__main__":
    main()