import math
import random

import numpy as np
import matplotlib.pyplot as plt

CELL = 0.05
MAP_SIZE = 4.0
N = int(MAP_SIZE / CELL)

ORIGIN_MIN = 2.0
ORIGIN_MAX = 2.5
RANGE_MIN = 0.3
RANGE_MAX = 1.5

UNKNOWN = 0.5
FREE = 0.0
OCCUPIED = 1.0


def to_cell(x, y):
    i = int(x / CELL)
    j = int(y / CELL)
    if i >= N:
        i = N - 1
    if j >= N:
        j = N - 1
    return i, j


def plot_line_low(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    d = 2 * dy - dx
    y = y0

    points = []
    for x in range(x0, x1 + 1):
        points.append((x, y))
        if d > 0:
            y += yi
            d += 2 * (dy - dx)
        else:
            d += 2 * dy
    return points


def plot_line_high(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    d = 2 * dx - dy
    x = x0

    points = []
    for y in range(y0, y1 + 1):
        points.append((x, y))
        if d > 0:
            x += xi
            d += 2 * (dx - dy)
        else:
            d += 2 * dx
    return points


def bresenham(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            points = plot_line_low(x1, y1, x0, y0)
            points.reverse()
        else:
            points = plot_line_low(x0, y0, x1, y1)
    else:
        if y0 > y1:
            points = plot_line_high(x1, y1, x0, y0)
            points.reverse()
        else:
            points = plot_line_high(x0, y0, x1, y1)
    return points


def make_scan():
    x0 = random.uniform(ORIGIN_MIN, ORIGIN_MAX)
    y0 = random.uniform(ORIGIN_MIN, ORIGIN_MAX)
    theta = math.radians(random.uniform(0, 360))
    r = random.uniform(RANGE_MIN, RANGE_MAX)

    x1 = min(max(x0 + r * math.cos(theta), 0.0), MAP_SIZE)
    y1 = min(max(y0 + r * math.sin(theta), 0.0), MAP_SIZE)

    i0, j0 = to_cell(x0, y0)
    i1, j1 = to_cell(x1, y1)
    path = bresenham(i0, j0, i1, j1)

    grid = np.full((N, N), UNKNOWN)
    for i, j in path[:-1]:
        grid[j, i] = FREE
    hi, hj = path[-1]
    grid[hj, hi] = OCCUPIED

    return grid, (x0, y0), (x1, y1)


def redraw(ax):
    grid, origin, hit = make_scan()

    ax.clear()
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1, origin="lower", extent=[0, MAP_SIZE, 0, MAP_SIZE])
    ax.plot([origin[0], hit[0]], [origin[1], hit[1]], "r-", linewidth=1.5)
    ax.plot(origin[0], origin[1], "ro", markersize=5)

    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(MAP_SIZE, 0)
    ax.set_xticks(np.arange(0, MAP_SIZE + 0.5, 0.5))
    ax.set_yticks(np.arange(0, MAP_SIZE + 0.5, 0.5))
    ax.set_xticks(np.arange(0, MAP_SIZE + CELL, CELL), minor=True)
    ax.set_yticks(np.arange(0, MAP_SIZE + CELL, CELL), minor=True)
    ax.grid(which="minor", color="#999999", linewidth=0.2)
    ax.grid(which="major", color="#333333", linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("laser scan - press any key to rescan, ESC to quit")
    ax.figure.canvas.draw_idle()


def on_key(event, ax):
    if event.key == "escape":
        plt.close(event.canvas.figure)
    else:
        redraw(ax)


def main():
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.mpl_connect("key_press_event", lambda e: on_key(e, ax))
    redraw(ax)
    plt.show()


if __name__ == "__main__":
    main()
