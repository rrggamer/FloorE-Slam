import argparse

from mcl_common import MCLConfig, run_version


CONFIG = MCLConfig(
    name="Version 2",
    n_particles=100,
    particle_motion_noise=True,
    capture_dir="captures/version_2",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCL Version 2")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args()
    run_version(CONFIG, steps=args.steps, show_steps=args.show_steps)
