import os
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========

WORLD_SIZE = 10.0           # world size : WORLD_SIZE x WORLD_SIZE metres
N_LANDMARK = 10             # number of landmarks
N_PARTICLE = 100            # number of particles
SENSOR_RANGE = 1.0          # sensor detection radius (m)
ROBOT_START = (5.0, 7.0, 45.0)  # initial x, y, heading angle (degrees)
MOVE_DISTANCE = 0.2         # commanded forward distance (m)
TURN_ANGLE = 10.0           # commanded left turn (degrees)
MOVE_STD = 0.04             # forward motion noise (m)
TURN_STD = 2.0              # turn noise (degrees)

SEED = 42                   # seed for random
CAPTURE_DIR = "captures"


# ==========================


# ====== WORLD Class =======

class MCLWorld:
    def __init__(
        self, size=WORLD_SIZE, seed=SEED, n_landmark=N_LANDMARK,
        robot_start=ROBOT_START,
    ):
        self.size = float(size)
        self.rng = np.random.default_rng(seed)
        self.bounds = np.array([0.0, self.size, 0.0, self.size])
        self.robot_x, self.robot_y, self.robot_heading = robot_start
        self.command_x, self.command_y, self.command_heading = robot_start

        self.landmarks = self.spawn_landmarks(n_landmark)
        self.detected_landmarks = self.detect_landmarks(
            self.robot_x, self.robot_y,
        )
        self.landmark_measurements = self.measure_landmarks()
        self.particles = self.spawn_particles(N_PARTICLE)

    @property
    def area(self):
        return self.size * self.size     # square metres

    def spawn_landmarks(self, n):
        """Return an (n, 2) array of random (x, y) landmark positions."""
        x_min, x_max, y_min, y_max = self.bounds
        xs = self.rng.uniform(x_min, x_max, size=n)
        ys = self.rng.uniform(y_min, y_max, size=n)
        return np.column_stack([xs, ys])

    def spawn_particles(self, n):
        """Return an (n, 3) array of random (x_p, y_p, zeta_p) particles."""
        x_min, x_max, y_min, y_max = self.bounds
        xs = self.rng.uniform(x_min, x_max, size=n)
        ys = self.rng.uniform(y_min, y_max, size=n)
        headings = self.rng.uniform(-180.0, 180.0, size=n)
        return np.column_stack([xs, ys, headings])

    def detect_landmarks(self, x, y):
        """Return landmark indices within SENSOR_RANGE of (x, y)."""
        robot_position = np.array([x, y])
        distances = np.linalg.norm(self.landmarks - robot_position, axis=1)
        return np.flatnonzero(distances <= SENSOR_RANGE)

    def measure_landmarks(self):
        """Return detected landmark measurements as (id, range, bearing)."""
        measurements = []
        robot_position = np.array([self.robot_x, self.robot_y])
        heading = self.robot_heading

        for landmark_id in self.detected_landmarks:
            landmark_position = self.landmarks[landmark_id]
            relative_position = landmark_position - robot_position
            distance = np.linalg.norm(relative_position)
            global_angle = np.rad2deg(np.arctan2(
                relative_position[1], relative_position[0],
            ))
            bearing = (global_angle - heading + 180.0) % 360.0 - 180.0
            measurements.append((landmark_id, distance, bearing))

        return np.array(measurements, dtype=float).reshape((-1, 3))

    def move_particles(self):
        """Move every particle using the ideal motion command without noise."""
        headings_rad = np.deg2rad(self.particles[:, 2])
        self.particles[:, 0] += MOVE_DISTANCE * np.cos(headings_rad)
        self.particles[:, 1] += MOVE_DISTANCE * np.sin(headings_rad)
        self.particles[:, 2] += TURN_ANGLE

        x_min, x_max, y_min, y_max = self.bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], x_min, x_max)
        self.particles[:, 1] = np.clip(self.particles[:, 1], y_min, y_max)

    def execute_motion(self):
        """Apply forward-then-left motion to ideal and noisy robot states."""
        heading_rad = np.deg2rad(self.command_heading)
        self.command_x += MOVE_DISTANCE * np.cos(heading_rad)
        self.command_y += MOVE_DISTANCE * np.sin(heading_rad)
        self.command_heading += TURN_ANGLE

        actual_distance = self.rng.normal(MOVE_DISTANCE, MOVE_STD)
        actual_turn = self.rng.normal(TURN_ANGLE, TURN_STD)
        heading_rad = np.deg2rad(self.robot_heading)
        self.robot_x += actual_distance * np.cos(heading_rad)
        self.robot_y += actual_distance * np.sin(heading_rad)
        self.robot_heading += actual_turn
        self.move_particles()

        self.detected_landmarks = self.detect_landmarks(
            self.robot_x, self.robot_y,
        )
        self.landmark_measurements = self.measure_landmarks()
        return actual_distance, actual_turn

# ==========================


# ======= Visualize ========

class MCLViz:
    def __init__(self, world):
        self.w = world
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def draw_world(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            ax.clear()

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
        self.draw_particles(ax)
        self.draw_robot(ax)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"World  {self.w.size:.0f} x {self.w.size:.0f} m  "
            f"(area = {self.w.area:.0f} m^2, landmarks = {len(self.w.landmarks)})"
        )
        # ax.legend(loc="upper right")

        return fig, ax

    def draw_landmarks(self, ax):
        lm = self.w.landmarks
        ax.scatter(
            lm[:, 0], lm[:, 1],
            marker="s", s=180, color="tab:red",
            edgecolors="black", linewidths=0.6, zorder=5,
            label="landmark",
        )
        detected = self.w.detected_landmarks
        if len(detected) > 0:
            ax.scatter(
                lm[detected, 0], lm[detected, 1],
                s=260, facecolors="none", edgecolors="tab:green",
                linewidths=2.0, zorder=6, label="detected landmark",
            )
        #for i, (x, y) in enumerate(lm):
        #    ax.annotate(f"L{i}", (x, y), textcoords="offset points",
        #                xytext=(6, 6), fontsize=8)

    def draw_particles(self, ax):
        """Draw particle positions without showing their headings."""
        particles = self.w.particles
        ax.scatter(
            particles[:, 0], particles[:, 1],
            s=18, color="tab:red", alpha=0.65,
            label="particle",
        )

    def draw_robot(self, ax):
        """Draw the robot position and an arrow showing its forward direction."""
        x = self.w.robot_x
        y = self.w.robot_y
        heading_rad = np.deg2rad(self.w.robot_heading)
        arrow_length = 0.8

        ax.scatter(
            x, y, s=180, color="tab:blue", edgecolors="black",
            linewidths=0.8, zorder=6, label="robot (true)",
        )
        sensor_circle = plt.Circle(
            (x, y), SENSOR_RANGE, fill=False,
            color="tab:green", linestyle="--", linewidth=1.5,
            label=f"sensor range ({SENSOR_RANGE:.1f} m)",
        )
        ax.add_patch(sensor_circle)
        ax.quiver(
            x, y,
            arrow_length * np.cos(heading_rad),
            arrow_length * np.sin(heading_rad),
            angles="xy", scale_units="xy", scale=1,
            color="tab:blue", width=0.008, zorder=7,
        )
        command_heading_rad = np.deg2rad(self.w.command_heading)
        ax.scatter(
            self.w.command_x, self.w.command_y, s=180,
            color="tab:green", edgecolors="black", linewidths=0.8,
            zorder=6, label="robot (command)",
        )
        ax.quiver(
            self.w.command_x, self.w.command_y,
            arrow_length * np.cos(command_heading_rad),
            arrow_length * np.sin(command_heading_rad),
            angles="xy", scale_units="xy", scale=1,
            color="tab:green", width=0.008, zorder=7,
        )

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
    # for i, (x, y) in enumerate(world.landmarks):
    #     print(f"  L{i:<2d} = ({x:6.3f}, {y:6.3f})")
    print(
        "ROBOT_START:",
        f"({world.robot_x:.3f}, {world.robot_y:.3f}, "
        f"heading={world.robot_heading:.1f} deg)",
    )
    print("SENSOR_RANGE:", SENSOR_RANGE, "m")
    print("DETECTED_LANDMARKS:", world.detected_landmarks.tolist())
    print("MEASUREMENTS (landmark_id, distance_m, bearing_deg):")
    for landmark_id, distance, bearing in world.landmark_measurements:
        print(f"  L{int(landmark_id)} = "
              f"(distance={distance:.3f}, bearing={bearing:.2f} deg)")
    print("PRESS ANY KEY IN THE FIGURE TO MOVE THE ROBOT")

    viz = MCLViz(world)
    fig, ax = viz.draw_world()
    motion_count = 0

    def on_key(event):
        nonlocal motion_count
        motion_count += 1
        actual_distance, actual_turn = world.execute_motion()
        viz.draw_world(fig, ax)
        fig.savefig(
            os.path.join(CAPTURE_DIR, "step10_particles.png"),
            dpi=150, bbox_inches="tight",
        )
        fig.canvas.draw_idle()
        print("MOTION_COUNT:", motion_count)
        print("ACTUAL_DISTANCE:", f"{actual_distance:.3f}", "m")
        print("ACTUAL_TURN:", f"{actual_turn:.2f}", "deg")
        print(
            "COMMAND_POSE:",
            f"({world.command_x:.3f}, {world.command_y:.3f}, "
            f"heading={world.command_heading:.2f} deg)",
        )
        print(
            "TRUE_POSE:",
            f"({world.robot_x:.3f}, {world.robot_y:.3f}, "
            f"heading={world.robot_heading:.2f} deg)",
        )
        print("STEP 11 MEASUREMENTS (landmark_id, distance_m, bearing_deg):")
        if len(world.landmark_measurements) == 0:
            print("  no landmark detected")
        else:
            for landmark_id, distance, bearing in world.landmark_measurements:
                print(
                    f"  L{int(landmark_id)} = "
                    f"(distance={distance:.3f}, bearing={bearing:.2f} deg)"
                )
        print("CAPTURE: captures/step10_particles.png")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
