import argparse

from mcl_common import MCLConfig, run_version


CONFIG = MCLConfig(
    name="Version 3 AMCL",
    n_particles=1000,
    particle_motion_noise=True,
    adaptive=True,
    min_particles=200,
    particle_decrement=10,
    capture_dir="captures/version_3",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCL Version 3: AMCL")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--show-steps", action="store_true")
    args = parser.parse_args()
    run_version(CONFIG, steps=args.steps, show_steps=args.show_steps)
