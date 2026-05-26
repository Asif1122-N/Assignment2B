from __future__ import annotations

import argparse

from routing import route_between
from train_models import main as train_main


def main():
    """Run training or route search from the command line."""

    parser = argparse.ArgumentParser(
        description="Traffic-based Route Guidance System"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train ML models
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lstm", "gru"],
        choices=["xgb", "lstm", "gru"],
        help="Choose models to train: xgb, lstm, gru"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for LSTM and GRU"
    )

    # Find routes
    route_parser = subparsers.add_parser("route")
    route_parser.add_argument("--origin", type=int, required=True)
    route_parser.add_argument("--destination", type=int, required=True)
    route_parser.add_argument("--datetime", required=True)
    route_parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "train":
        train_main(args.models, args.epochs)

    if args.command == "route":
        routes = route_between(
            args.origin,
            args.destination,
            args.datetime,
            args.k
        )

        for number, route in enumerate(routes, start=1):
            print(
                f"Route {number}: {route['route']} "
                f"- {route['estimated_minutes']} minutes"
            )


if __name__ == "__main__":
    main()