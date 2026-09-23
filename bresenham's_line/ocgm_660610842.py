import csv
import math

import numpy as np
import matplotlib.pyplot as plt

N_BEAMS = 11
CELL = 1.0

L_OCC = 0.85
L_FREE = -0.4
L_MAX = 10.0

OCC_THRESH = 0.6
FREE_THRESH = 0.4


def load_scans(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [list(map(float, row)) for row in reader]
    return np.array(rows)


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


def compute_bounds(data, n_beams):
    xs = [data[:, 0]]
    ys = [data[:, 1]]
    th = data[:, 2]
    for k in range(n_beams):
        z = data[:, 3 + 2 * k]
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


def apply_scan(log_odds, row, x_min, y_min, width, height, n_beams=N_BEAMS, cell=CELL):
    rx, ry, rth = row[0], row[1], row[2]
    i0 = min(max(int((rx - x_min) / cell), 0), width - 1)
    j0 = min(max(int((ry - y_min) / cell), 0), height - 1)

    for k in range(n_beams):
        z = row[3 + 2 * k]
        phi = row[4 + 2 * k]
        if math.isnan(z) or z <= 0:
            continue

        ang = rth + phi
        hx = rx + z * math.cos(ang)
        hy = ry + z * math.sin(ang)
        i1 = min(max(int((hx - x_min) / cell), 0), width - 1)
        j1 = min(max(int((hy - y_min) / cell), 0), height - 1)

        path = bresenham(i0, j0, i1, j1)
        for i, j in path[:-1]:
            log_odds[j, i] += L_FREE
        hi, hj = path[-1]
        log_odds[hj, hi] += L_OCC


def beam_endpoints(row, n_beams=N_BEAMS):
    rx, ry, rth = row[0], row[1], row[2]
    ends = []
    for k in range(n_beams):
        z = row[3 + 2 * k]
        phi = row[4 + 2 * k]
        if math.isnan(z) or z <= 0:
            continue
        ang = rth + phi
        ends.append((rx + z * math.cos(ang), ry + z * math.sin(ang)))
    return rx, ry, ends


def to_display(log_odds):
    prob = 1.0 - 1.0 / (1.0 + np.exp(np.clip(log_odds, -L_MAX, L_MAX)))
    disp = np.full(prob.shape, 0.5)
    disp[prob >= OCC_THRESH] = 1.0
    disp[prob <= FREE_THRESH] = 0.0
    return disp


def redraw(ax, log_odds, data, step, bounds):
    x_min, x_max, y_min, y_max = bounds
    disp = to_display(log_odds)

    ax.clear()
    ax.imshow(disp, cmap="gray_r", vmin=0, vmax=1, origin="lower",
              extent=[x_min, x_max, y_min, y_max], interpolation="nearest")

    done = data[:step]
    if len(done):
        ax.plot(done[:, 0], done[:, 1], "b-", linewidth=0.8, alpha=0.5)

    if step > 0:
        rx, ry, ends = beam_endpoints(data[step - 1])
        for hx, hy in ends:
            ax.plot([rx, hx], [ry, hy], color="red", linewidth=0.8, alpha=0.7)
        ax.plot(rx, ry, "o", color="red", markersize=8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"scan {step}/{len(data)}  |  press any key = next scan  |  ESC = quit")
    ax.figure.canvas.draw_idle()


def on_key(event, ax, log_odds, data, state, bounds):
    if event.key == "escape":
        plt.close(event.canvas.figure)
        return
    if state["step"] < len(data):
        apply_scan(log_odds, data[state["step"]], *bounds_wh(bounds))
        state["step"] += 1
    redraw(ax, log_odds, data, state["step"], bounds)


def bounds_wh(bounds):
    x_min, x_max, y_min, y_max = bounds
    width = int((x_max - x_min) / CELL)
    height = int((y_max - y_min) / CELL)
    return x_min, y_min, width, height


def main():
    data = load_scans("data.csv")
    bounds = compute_bounds(data, N_BEAMS)
    x_min, y_min, width, height = bounds_wh(bounds)
    log_odds = np.zeros((height, width))
    state = {"step": 0}

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.canvas.mpl_connect("key_press_event", lambda e: on_key(e, ax, log_odds, data, state, bounds))
    redraw(ax, log_odds, data, state["step"], bounds)
    plt.show()


if __name__ == "__main__":
    main()
