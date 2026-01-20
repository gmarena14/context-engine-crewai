# src/run.py
from __future__ import annotations

import argparse
from pathlib import Path

from data_simulator import SimConfig, generate_items, generate_search_events, save_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_data", action="store_true", help="Generate synthetic items + events")
    args = parser.parse_args()

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    if args.generate_data:
        cfg = SimConfig()
        items = generate_items(cfg)
        events = generate_search_events(cfg, items)
        save_artifacts(items, events, out_dir=str(artifacts))
        print("✅ Data generated under artifacts/")
    else:
        print("Run: python src/run.py --generate_data")


if __name__ == "__main__":
    main()
