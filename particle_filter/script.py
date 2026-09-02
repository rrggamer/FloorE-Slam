import os
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========

WORLD_SIZE = 10.0           # world size : WORLD_SIZE x WORLD_SIZE metres
N_LANDMARK = 10             # number of landmarks
N_PARTICLE = 100            # number of particles
SENSOR_RANGE = 1.0          # sensor detection radius (m)

SEED = 42                   # seed for random
CAPTURE_DIR = "captures"


# ==========================


# ====== WORLD Class =======

class MCLWorld:
    def __init__(self, size=WORLD_SIZE, seed=SEED, n_landmark=N_LANDMARK):
        self.size = float(size)
        self.rng = np.random.default_rng(seed)
        self.bounds = np.array([0.0, self.size, 0.0, self.size])

        
        self.landmarks = self.spawn_landmarks(n_landmark)

    @property
    def area(self):
        return self.size * self.size     # square metres

    def spawn_landmarks(self, n):
        """Return an (n, 2) array of random (x, y) landmark positions."""
        x_min, x_max, y_min, y_max = self.bounds
        xs = self.rng.uniform(x_min, x_max, size=n)
        ys = self.rng.uniform(y_min, y_max, size=n)
        return np.column_stack([xs, ys])

# ==========================


# ======= Visualize ========

class MCLViz:
    def __init__(self, world):
        self.w = world
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def draw_world(self):
        fig, ax = plt.subplots(figsize=(6, 6))

        x_min, x_max, y_min, y_max = self.w.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")

        # border of the free space
        ax.plot(
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color="black", linewidth=2,
        )

        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.grid(True, linestyle='--', alpha=0.4)

        self.draw_landmarks(ax)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"World  {self.w.size:.0f} x {self.w.size:.0f} m  "
            f"(area = {self.w.area:.0f} m^2, landmarks = {len(self.w.landmarks)})"
        )
        ax.legend(loc="upper right")

        return fig, ax

    def draw_landmarks(self, ax):
        lm = self.w.landmarks
        ax.scatter(
            lm[:, 0], lm[:, 1],
            marker="s", s=180, color="tab:red",
            edgecolors="black", linewidths=0.6, zorder=5,
            label="landmark",
        )
        #for i, (x, y) in enumerate(lm):
        #    ax.annotate(f"L{i}", (x, y), textcoords="offset points",
        #                xytext=(6, 6), fontsize=8)

    def capture(self, name, show=False):
        fig, _ = self.draw_world()
        path = os.path.join(CAPTURE_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()          # opens the window, blocks until you close it
        plt.close(fig)
        return path

# ==========================


def main():
    world = MCLWorld()

    print("SEED:", SEED)
    print("RNG:", world.rng)
    print("WORLD_SIZE:", world.size, "m")
    print("BOUNDS:", world.bounds.tolist(), "[x_min, x_max, y_min, y_max]")
    print("AREA:", world.area, "m^2")
    print("N_LANDMARK:", len(world.landmarks))
    for i, (x, y) in enumerate(world.landmarks):
        print(f"  L{i:<2d} = ({x:6.3f}, {y:6.3f})")

    viz = MCLViz(world)
    out = viz.capture("step2_landmarks.png", show=True)
    print("CAPTURE:", out)


if __name__ == "__main__":
    main()
