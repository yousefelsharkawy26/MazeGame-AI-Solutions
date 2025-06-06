# Maze Solver Project Summary

This project is a Python-based maze-solving visualizer that implements and compares **five pathfinding algorithms** to find a path from a start point to an end point in a maze.

---

## ✅ Implemented Algorithms

### 1. Breadth-First Search (BFS)

- Explores all neighboring nodes level by level.
- **Guarantees the shortest path** (in unweighted graphs).
- Works well and is fast for small to medium mazes.

### 2. Depth-First Search (DFS)

- Explores as deep as possible before backtracking.
- May not find the shortest path.
- Can get stuck in deep branches in complex mazes.

### 3. Uniform Cost Search (UCS)

- Like BFS but considers move **costs**.
- Expands the least-cost node first.
- Guarantees the lowest-cost path (if costs are different).

### 4. Greedy Best-First Search

- Uses **heuristic (estimated distance to goal)** only.
- Fast, but not always optimal.
- Can get stuck if the heuristic misleads it.

### 5. Genetic Algorithm (GA)

- Inspired by natural evolution.
- Generates and evolves a population of random paths using **fitness, crossover, mutation**.
- May take longer and **does not guarantee optimal path**, but shows evolutionary approach in action.

---

## 🎯 Purpose

- Compare how each algorithm behaves.
- Visualize how they explore and find paths in a maze.
- Learn the trade-offs between **optimality**, **speed**, and **exploration strategy**.

---

## 🤩 Features

- Step-by-step visualization of algorithms.
- Support for both weighted and unweighted mazes.
- Random maze generation.
- Easy switching between algorithms via interface.
