#!/usr/bin/env python3
"""Build a 2D occupancy grid map (OCGM) from data.csv recorded by record_data.py.

Ported from ocgm_660610842.py (Bresenham + log-odds, same L_OCC/L_FREE/L_MAX
constants) and generalized from that assignment's 11-beam integer grid-world
data to this one's 360-beam, meter-scale, continuous-coordinate lidar data:
  - n_beams is read from the CSV width instead of hardcoded.
  - a beam at (or past) max_range means "no obstacle in range", so only the
    free-space carve happens for it -- no occupied endpoint is marked
    (ocgm_660610842.py's data never had this case, since its sim always
    returned a real hit).
  - CELL is a real grid resolution in meters (default 0.1), not 1 grid-unit.

This is a standalone script (run with plain python3, not through ROS/colcon).

CSV row layout (same as record_data.py):
    col 0        = X  (sensor base position, meters)
    col 1        = Y  (sensor base position, meters)
    col 2        = TH (sensor base heading, radians)
    col 3+2*i    = range of ray i (meters)
    col 4+2*i    = angle of ray i relative to the sensor's forward axis (radians)

Usage:
    pip install numpy matplotlib
    python3 build_ocgm.py data.csv --resolution 0.1 --out ogcm_xxx.csv
    python3 build_ocgm.py data.csv --no-interactive   # skip the live viewer
"""
import argparse
import csv
import math

import matplotlib.pyplot as plt
import numpy as np

L_OCC = 0.85
L_FREE = -0.4
L_MAX = 10.0

OCC_THRESH = 0.6
FREE_THRESH = 0.4


def load_scans(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    try:
        float(rows[0][0])
    except ValueError:
        rows = rows[1:]  # header row, like ocgm_660610842.py's data.csv has
    data = np.array([[float(v) for v in row] for row in rows])
    n_beams = (data.shape[1] - 3) // 2
    return data, n_beams


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


def compute_bounds(data, n_beams, max_range):
    xs = [data[:, 0]]
    ys = [data[:, 1]]
    th = data[:, 2]
    for k in range(n_beams):
        z = np.minimum(data[:, 3 + 2 * k], max_range)
        phi = data[:, 4 + 2 * k]
        valid = ~np.isnan(z) & (z > 0)
        ang = th[valid] + phi[valid]
        xs.append(data[valid, 0] + z[valid] * np.cos(ang))
        ys.append(data[valid, 1] + z[valid] * np.sin(ang))
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    x_min = math.floor(min(0.0, all_x.min()))
    y_min = math.floor(min(0.0, all_y.min()))
    x_max = math.ceil(all_x.max()) + 1
    y_max = math.ceil(all_y.max()) + 1
    return x_min, x_max, y_min, y_max


def bounds_wh(bounds, cell):
    x_min, x_max, y_min, y_max = bounds
    width = int((x_max - x_min) / cell)
    height = int((y_max - y_min) / cell)
    return x_min, y_min, width, height


def apply_scan(log_odds, row, x_min, y_min, width, height, n_beams, cell, max_range):
    rx, ry, rth = row[0], row[1], row[2]
    i0 = min(max(int((rx - x_min) / cell), 0), width - 1)
    j0 = min(max(int((ry - y_min) / cell), 0), height - 1)

    for k in range(n_beams):
        z = row[3 + 2 * k]
        phi = row[4 + 2 * k]
        if math.isnan(z) or z <= 0:
            continue

        hit = z < max_range - 1e-3
        z = min(z, max_range)
        ang = rth + phi
        hx = rx + z * math.cos(ang)
        hy = ry + z * math.sin(ang)
        i1 = min(max(int((hx - x_min) / cell), 0), width - 1)
        j1 = min(max(int((hy - y_min) / cell), 0), height - 1)

        path = bresenham(i0, j0, i1, j1)
        for i, j in path[:-1]:
            log_odds[j, i] += L_FREE
        if hit:
            hi, hj = path[-1]
            log_odds[hj, hi] += L_OCC


def beam_endpoints(row, n_beams, max_range):
    rx, ry, rth = row[0], row[1], row[2]
    ends = []
    for k in range(n_beams):
        z = row[3 + 2 * k]
        phi = row[4 + 2 * k]
        if math.isnan(z) or z <= 0 or z >= max_range - 1e-3:
            continue
        ang = rth + phi
        ends.append((rx + z * math.cos(ang), ry + z * math.sin(ang)))
    return rx, ry, ends


def prob_grid(log_odds):
    return 1.0 - 1.0 / (1.0 + np.exp(np.clip(log_odds, -L_MAX, L_MAX)))


def to_display(log_odds):
    prob = prob_grid(log_odds)
    disp = np.full(prob.shape, 0.5)
    disp[prob >= OCC_THRESH] = 1.0
    disp[prob <= FREE_THRESH] = 0.0
    return disp


def redraw(ax, log_odds, data, step, bounds, step_scans):
    x_min, x_max, y_min, y_max = bounds
    disp = to_display(log_odds)

    ax.clear()
    ax.imshow(disp, cmap="gray_r", vmin=0, vmax=1, origin="lower",
              extent=[x_min, x_max, y_min, y_max], interpolation="nearest")

    done = data[:step]
    if len(done):
        ax.plot(done[:, 0], done[:, 1], "b-", linewidth=0.8, alpha=0.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"scan {step}/{len(data)}  |  key = +{step_scans} scans  |  ESC = stop & save")
    ax.figure.canvas.draw_idle()


def on_key(event, ax, log_odds, data, state, bounds, n_beams, cell, max_range, step_scans):
    if event.key == "escape":
        plt.close(event.canvas.figure)
        return

    x_min, y_min, width, height = bounds_wh(bounds, cell)
    end = min(state["step"] + step_scans, len(data))
    while state["step"] < end:
        apply_scan(log_odds, data[state["step"]], x_min, y_min, width, height, n_beams, cell, max_range)
        state["step"] += 1

    redraw(ax, log_odds, data, state["step"], bounds, step_scans)
    if state["step"] >= len(data):
        plt.close(event.canvas.figure)


def compute_hits(data, n_beams, max_range):
    """All ray endpoints that actually hit something, world-frame -- used
    only for the final top-view comparison plot, not for the grid itself."""
    th = data[:, 2]
    hit_x, hit_y = [], []
    for k in range(n_beams):
        z = data[:, 3 + 2 * k]
        phi = data[:, 4 + 2 * k]
        valid = ~np.isnan(z) & (z > 0) & (z < max_range - 1e-3)
        ang = th[valid] + phi[valid]
        hit_x.append(data[valid, 0] + z[valid] * np.cos(ang))
        hit_y.append(data[valid, 1] + z[valid] * np.sin(ang))
    return np.concatenate(hit_x), np.concatenate(hit_y)


def plot_comparison(data, grid, bounds, cell, hit_x, hit_y):
    x_min, x_max, y_min, y_max = bounds
    extent = [x_min, x_max, y_min, y_max]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(1.0 - grid, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=1)
    axes[0].set_title("Occupancy Grid Map")

    axes[1].scatter(hit_x, hit_y, s=0.5, color="blue")
    axes[1].plot(data[:, 0], data[:, 1], color="red", linewidth=0.8, label="robot path")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Top View (accumulated lidar hits + trajectory)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    plt.tight_layout()
    plt.savefig("ocgm_comparison.png", dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_csv", nargs="?", default="data.csv")
    parser.add_argument("--resolution", type=float, default=0.1, help="grid cell size in meters")
    parser.add_argument("--max-range", type=float, default=11.9,
                         help="lidar max range in meters (just under the sensor's 12.0m limit)")
    parser.add_argument("--out", default="ogcm.csv", help="output occupancy grid CSV path")
    parser.add_argument("--step", type=int, default=20, help="scans applied per keypress in the live viewer")
    parser.add_argument("--no-interactive", action="store_true",
                         help="skip the live viewer, batch-process every scan immediately")
    args = parser.parse_args()

    data, n_beams = load_scans(args.data_csv)
    bounds = compute_bounds(data, n_beams, args.max_range)
    x_min, y_min, width, height = bounds_wh(bounds, args.resolution)
    log_odds = np.zeros((height, width))

    if args.no_interactive:
        for row in data:
            apply_scan(log_odds, row, x_min, y_min, width, height, n_beams, args.resolution, args.max_range)
    else:
        print(f"Press any key to apply {args.step} more scans, ESC to stop early and save.")
        state = {"step": 0}
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: on_key(e, ax, log_odds, data, state, bounds, n_beams,
                              args.resolution, args.max_range, args.step),
        )
        redraw(ax, log_odds, data, state["step"], bounds, args.step)
        plt.show()

    grid = prob_grid(log_odds)
    np.savetxt(args.out, grid, delimiter=",")
    print(f"Saved {grid.shape[1]}x{grid.shape[0]} occupancy grid to {args.out}")

    hit_x, hit_y = compute_hits(data, n_beams, args.max_range)
    plot_comparison(data, grid, bounds, args.resolution, hit_x, hit_y)


if __name__ == "__main__":
    main()
