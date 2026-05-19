from __future__ import annotations

import argparse

from routing import route_between
from train_models import main as train_main


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line menu for the Traffic-based Route Guidance System."""

    parser = argparse.ArgumentParser(
        description="Traffic-based Route Guidance System for traffic prediction and route search"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train machine learning models for traffic prediction"
    )

    train_parser.add_argument(
        "--models",
        nargs="+",
        default=["xgb", "lstm", "gru"],
        choices=["rf", "xgb", "lstm", "gru"],
        help="Choose which models to train: rf, xgb, lstm, gru"
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs for LSTM and GRU"
    )

    # Route command
    route_parser = subparsers.add_parser(
        "route",
        help="Find the top-k routes between two SCATS intersections"
    )

    route_parser.add_argument(
        "--origin",
        type=int,
        required=True,
        help="Origin SCATS site number, for example 2000"
    )

    route_parser.add_argument(
        "--destination",
        type=int,
        required=True,
        help="Destination SCATS site number, for example 3002"
    )

    route_parser.add_argument(
        "--datetime",
        required=True,
        help='Date and time for prediction, for example "2006-10-18 08:15"'
    )

    route_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of routes to return"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_main(args.models, args.epochs)

    elif args.command == "route":
        routes = route_between(
            args.origin,
            args.destination,
            args.datetime,
            args.k
        )

        for index, item in enumerate(routes, start=1):
            print(
                f"Route {index}: {item['route']} "
                f"- {item['estimated_minutes']} minutes"
            )


if __name__ == "__main__":
    main()