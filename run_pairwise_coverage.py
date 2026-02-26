from __future__ import annotations

import argparse
from pathlib import Path

from pairwise_coverage_app.app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone pairwise image voting app")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/benchmark/Images"),
        help="Directory containing images used for pairwise voting",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Minimum number of times each image must be shown globally",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("pairwise_coverage_app/state"),
        help="Directory used for persistent CSV and state files",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5050, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(data_dir=args.data_dir, target_n=args.n, state_dir=args.state_dir)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
