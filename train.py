import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "release" / "USTMT_Best.py"


def main():
    parser = argparse.ArgumentParser(
        description="Launch UST-MT-DeepLabv3+ training with a selected config."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the experiment config file.",
    )
    args, passthrough = parser.parse_known_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cmd = [
        sys.executable,
        "-m",
        "semi_supervised_segmentation.main",
        "--config",
        str(config_path),
    ]
    cmd.extend(passthrough)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()

