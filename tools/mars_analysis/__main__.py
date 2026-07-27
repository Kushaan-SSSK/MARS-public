from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import AnalysisConfig, default_config
from .metrics import validate_predictions
from .pipeline import AnalysisRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mars_analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-config", help="Write a starter analysis configuration")
    init.add_argument("--output", default="analysis_config.json")
    tree = sub.add_parser("build-edf-tree-config", help="Build an analysis configuration from an EDF folder")
    tree.add_argument("--edf-root", required=True)
    tree.add_argument("--output", default="analysis_config.edf_tree.json")
    tree.add_argument("--output-root", default=None, help="Folder where score and review outputs will be written")
    tree.add_argument("--limit", type=int, default=None)
    tree.add_argument("--dataset-id", default="edf_local")
    for command in ["inventory", "analyze", "gui"]:
        item = sub.add_parser(command)
        item.add_argument("config", nargs="?", default="analysis_config.json")
    validate = sub.add_parser("validate", help="Compare MARS predictions with your own epoch labels")
    validate.add_argument("--predictions", required=True)
    validate.add_argument("--labels", required=True)
    validate.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "init-config":
        default_config().save_json(args.output)
        print(f"Wrote {Path(args.output).resolve()}")
        return 0
    if args.command == "build-edf-tree-config":
        from .manifest_config import build_config_from_edf_tree
        cfg = build_config_from_edf_tree(
            args.edf_root, output_path=args.output, output_root=args.output_root,
            limit=args.limit, dataset_id=args.dataset_id,
        )
        print(f"Wrote {Path(args.output).resolve()} with {len(cfg.recordings)} recordings")
        return 0
    if args.command == "validate":
        summary, confusion = validate_predictions(args.predictions, args.labels, args.output_dir)
        print(f"Validated {int(summary['AlignedEpochs'].iloc[0]) if not summary.empty else 0} aligned epochs -> {Path(args.output_dir).resolve()}")
        return 0
    if args.command == "gui":
        from .gui import run_gui
        return run_gui(args.config)
    runner = AnalysisRunner(AnalysisConfig.from_json(args.config))
    if args.command == "inventory":
        rows = runner.run_inventory()
        print(f"Inventoried {len(rows)} label rows -> {runner.output_dir / 'label_inventory.csv'}")
    else:
        for name, path in runner.run_analysis().items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
