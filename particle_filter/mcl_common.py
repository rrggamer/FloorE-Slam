import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MCLConfig:
    name: str
    n_particles: int
    particle_motion_noise: bool
    adaptive: bool = False
    min_particles: int = 200
    particle_decrement: int = 10
    range_std: float = 0.10
    bearing_std: float = 5.0
    world_size: float = 10.0
    n_landmarks: int = 10
    sensor_range: float = 1.0
    robot_start: tuple = (5.0, 7.0, 45.0)
    move_distance: float = 0.2
    turn_angle: float = 10.0
    move_std: float = 0.04
    turn_std: float = 2.0
    seed: int = 42
    capture_dir: str = "captures"


class MCLWorld:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.size = float(config.world_size)
        self.bounds = np.array([0.0, self.size, 0.0, self.size])
        self.robot_x, self.robot_y, self.robot_heading = config.robot_start
        self.command_x, self.command_y, self.command_heading = config.robot_start
        self.landmarks = self.spawn_landmarks(config.n_landmarks)
        self.detected_landmarks = self.detect_landmarks(
            self.robot_x, self.robot_y,
        )
        self.landmark_measurements = self.measure_landmarks()
        self.particles = self.spawn_particles(config.n_particles)
        self.weights = np.full(len(self.particles), 1.0 / len(self.particles))
        self.motion_count = 0

    @property
    def area(self):
        return self.size * self.size

    def spawn_landmarks(self, count):
        x_min, x_max, y_min, y_max = self.bounds
        return np.column_stack((
            self.rng.uniform(x_min, x_max, size=count),
            self.rng.uniform(y_min, y_max, size=count),
        ))

    def spawn_particles(self, count):
        x_min, x_max, y_min, y_max = self.bounds
        return np.column_stack((
            self.rng.uniform(x_min, x_max, size=count),
            self.rng.uniform(y_min, y_max, size=count),
            self.rng.uniform(-180.0, 180.0, size=count),
        ))

    def detect_landmarks(self, x, y):
        distances = np.linalg.norm(self.landmarks - np.array([x, y]), axis=1)
        return np.flatnonzero(distances <= self.config.sensor_range)

    def measure_landmarks(self):
        measurements = []
        robot_position = np.array([self.robot_x, self.robot_y])
        for landmark_id in self.detected_landmarks:
            relative = self.landmarks[landmark_id] - robot_position
            distance = np.linalg.norm(relative)
            global_angle = np.rad2deg(np.arctan2(relative[1], relative[0]))
            bearing = self.angle_difference(global_angle, self.robot_heading)
            measurements.append((landmark_id, distance, bearing))
        return np.array(measurements, dtype=float).reshape((-1, 3))

    @staticmethod
    def angle_difference(angle_a, angle_b):
        return (angle_a - angle_b + 180.0) % 360.0 - 180.0

    def move_particles(self):
        config = self.config
        if config.particle_motion_noise:
            distances = self.rng.normal(
                config.move_distance, config.move_std, len(self.particles),
            )
            turns = self.rng.normal(
                config.turn_angle, config.turn_std, len(self.particles),
            )
        else:
            distances = np.full(len(self.particles), config.move_distance)
            turns = np.full(len(self.particles), config.turn_angle)

        headings_rad = np.deg2rad(self.particles[:, 2])
        self.particles[:, 0] += distances * np.cos(headings_rad)
        self.particles[:, 1] += distances * np.sin(headings_rad)
        self.particles[:, 2] += turns
        self.clip_particles()

    def clip_particles(self):
        x_min, x_max, y_min, y_max = self.bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], x_min, x_max)
        self.particles[:, 1] = np.clip(self.particles[:, 1], y_min, y_max)
        self.particles[:, 2] = (self.particles[:, 2] + 180.0) % 360.0 - 180.0

    def execute_motion(self):
        config = self.config
        command_heading_rad = np.deg2rad(self.command_heading)
        self.command_x += config.move_distance * np.cos(command_heading_rad)
        self.command_y += config.move_distance * np.sin(command_heading_rad)
        self.command_heading += config.turn_angle

        actual_distance = self.rng.normal(config.move_distance, config.move_std)
        actual_turn = self.rng.normal(config.turn_angle, config.turn_std)
        heading_rad = np.deg2rad(self.robot_heading)
        self.robot_x += actual_distance * np.cos(heading_rad)
        self.robot_y += actual_distance * np.sin(heading_rad)
        self.robot_heading += actual_turn
        self.move_particles()

        self.detected_landmarks = self.detect_landmarks(self.robot_x, self.robot_y)
        self.landmark_measurements = self.measure_landmarks()
        self.update_weights()
        self.resample()
        self.motion_count += 1
        return actual_distance, actual_turn

    def predicted_measurement(self, particle, landmark_id):
        relative = self.landmarks[landmark_id] - particle[:2]
        distance = np.linalg.norm(relative)
        global_angle = np.rad2deg(np.arctan2(relative[1], relative[0]))
        bearing = self.angle_difference(global_angle, particle[2])
        return distance, bearing

    def update_weights(self):
        count = len(self.particles)
        if len(self.landmark_measurements) == 0:
            self.weights = np.full(count, 1.0 / count)
            return

        range_std = self.config.range_std
        bearing_std = self.config.bearing_std
        normal_range = range_std * np.sqrt(2.0 * np.pi)
        normal_bearing = bearing_std * np.sqrt(2.0 * np.pi)
        weights = np.zeros(count)

        for landmark_id, measured_range, measured_bearing in self.landmark_measurements:
            landmark_id = int(landmark_id)
            predicted = np.array([
                self.predicted_measurement(particle, landmark_id)
                for particle in self.particles
            ])
            range_error = predicted[:, 0] - measured_range
            bearing_error = self.angle_difference(
                predicted[:, 1], measured_bearing,
            )
            range_likelihood = np.exp(
                -0.5 * (range_error / range_std) ** 2
            ) / normal_range
            bearing_likelihood = np.exp(
                -0.5 * (bearing_error / bearing_std) ** 2
            ) / normal_bearing
            weights += range_likelihood + bearing_likelihood

        total = np.sum(weights)
        if not np.isfinite(total) or total <= 0.0:
            self.weights = np.full(count, 1.0 / count)
        else:
            self.weights = weights / total

    def resample(self):
        current_count = len(self.particles)
        target_count = current_count
        if self.config.adaptive:
            target_count = max(
                self.config.min_particles,
                current_count - self.config.particle_decrement,
            )

        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        positions = (
            self.rng.random() + np.arange(target_count)
        ) / target_count
        indexes = np.searchsorted(cumulative, positions)
        self.particles = self.particles[indexes].copy()
        self.weights = np.full(target_count, 1.0 / target_count)


class MCLViz:
    def __init__(self, world):
        self.world = world
        os.makedirs(world.config.capture_dir, exist_ok=True)

    def draw_world(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            ax.clear()

        x_min, x_max, y_min, y_max = self.world.bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.plot(
            [x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min],
            color="black", linewidth=2,
        )
        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.grid(True, linestyle="--", alpha=0.4)

        landmarks = self.world.landmarks
        ax.scatter(
            landmarks[:, 0], landmarks[:, 1], marker="s", s=180,
            color="tab:red", edgecolors="black", linewidths=0.6,
        )
        detected = self.world.detected_landmarks
        if len(detected) > 0:
            ax.scatter(
                landmarks[detected, 0], landmarks[detected, 1], s=260,
                facecolors="none", edgecolors="tab:green", linewidths=2.0,
            )

        particles = self.world.particles
        ax.scatter(particles[:, 0], particles[:, 1], s=18, color="tab:red", alpha=0.65)
        self.draw_pose(ax)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"{self.world.config.name} | particles={len(particles)} | "
            f"step={self.world.motion_count}"
        )
        return fig, ax

    def draw_pose(self, ax):
        world = self.world
        heading_rad = np.deg2rad(world.robot_heading)
        ax.scatter(world.robot_x, world.robot_y, s=180, color="tab:blue", edgecolors="black")
        ax.add_patch(plt.Circle(
            (world.robot_x, world.robot_y), world.config.sensor_range,
            fill=False, color="tab:green", linestyle="--", linewidth=1.5,
        ))
        ax.quiver(
            world.robot_x, world.robot_y, 0.8 * np.cos(heading_rad),
            0.8 * np.sin(heading_rad), angles="xy", scale_units="xy",
            scale=1, color="tab:blue", width=0.008,
        )

    def save(self, filename):
        fig, _ = self.draw_world()
        path = os.path.join(self.world.config.capture_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


def print_state(world):
    print("VERSION:", world.config.name)
    print("SEED:", world.config.seed)
    print("N_LANDMARK:", len(world.landmarks))
    print("INITIAL_PARTICLES:", world.config.n_particles)
    print("SENSOR_RANGE:", world.config.sensor_range, "m")
    print("DETECTED_LANDMARKS:", world.detected_landmarks.tolist())


def run_version(config, steps=0, show=True, show_steps=False):
    world = MCLWorld(config)
    viz = MCLViz(world)
    print_state(world)

    if steps > 0:
        fig = ax = None
        if show_steps:
            fig, ax = viz.draw_world()
            print("Press any key in the figure to run the next step.")

        for step in range(steps):
            if show_steps and not plt.waitforbuttonpress():
                break
            distance, turn = world.execute_motion()
            filename = f"{config.name.lower().replace(' ', '_')}_step_{step + 1:02d}.png"
            if show_steps:
                viz.draw_world(fig, ax)
                path = os.path.join(config.capture_dir, filename)
                fig.savefig(path, dpi=150, bbox_inches="tight")
                fig.canvas.draw_idle()
            else:
                path = viz.save(filename)
            print(
                f"STEP {step + 1}: particles={len(world.particles)}, "
                f"distance={distance:.3f} m, turn={turn:.2f} deg, capture={path}"
            )
        if show_steps:
            plt.close(fig)
        return

    fig, ax = viz.draw_world()
    print("Press any key in the figure to move the robot, or close the window.")

    def on_key(event):
        distance, turn = world.execute_motion()
        viz.draw_world(fig, ax)
        filename = f"{config.name.lower().replace(' ', '_')}_step_{world.motion_count:02d}.png"
        path = os.path.join(config.capture_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        fig.canvas.draw_idle()
        print(
            f"STEP {world.motion_count}: particles={len(world.particles)}, "
            f"distance={distance:.3f} m, turn={turn:.2f} deg, capture={path}"
        )

    fig.canvas.mpl_connect("key_press_event", on_key)
    if show:
        plt.show()
    plt.close(fig)
