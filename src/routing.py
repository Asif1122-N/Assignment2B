"""Route guidance integration layer.

This file provides a working end-to-end route search using NetworkX. When you
add your Assignment 2A code, the important integration point is the same:
replace each edge's static cost with predicted travel time, then run the top-k
path algorithm from Part A.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from config import CLEANED_DATA, TIME_SERIES_DATA
from data_processing import get_site_locations, load_time_series
from travel_time import edge_travel_time_minutes, haversine_km


def build_knn_graph(cleaned_csv_path: str | Path = CLEANED_DATA, neighbours: int = 3) -> nx.Graph:
    """replacing the original graph construction with a simple KNN graph based on geographic distance.
    This is a more realistic starting point for route search, as it reflects the actual road network more closely than a fully connected graph. You can adjust the number of neighbours to balance connectivity and practicality.
    """
    locs = get_site_locations(cleaned_csv_path)
    graph = nx.Graph()

    for _, row in locs.iterrows():
        graph.add_node(
            int(row["SCATS Number"]),
            label=row["Location"],
            lat=float(row["NB_LATITUDE"]),
            lon=float(row["NB_LONGITUDE"]),
        )

    coords = locs[["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]].to_numpy()
    for site, lat, lon in coords:
        distances = []
        for other, lat2, lon2 in coords:
            if int(site) == int(other):
                continue
            d = haversine_km(float(lat), float(lon), float(lat2), float(lon2))
            distances.append((d, int(other)))
        for d, other in sorted(distances)[:neighbours]:
            graph.add_edge(int(site), int(other), distance_km=float(d), travel_time=0.0)

    return graph


def historical_average_flow(df: pd.DataFrame, scats_number: int, when: datetime) -> float:
    """Simple fallback predictor: average traffic for this site, hour and minute."""
    site_df = df[df["SCATS Number"] == int(scats_number)].copy()
    same_slot = site_df[(site_df["Hour"] == when.hour) & (site_df["Minute"] == when.minute)]
    if len(same_slot):
        return float(same_slot["Traffic"].mean())
    return float(site_df["Traffic"].mean())


def update_edge_weights_with_flows(graph: nx.Graph, when: datetime, flow_df: pd.DataFrame) -> nx.Graph:
    """Update every edge cost using predicted/historical flow from the starting node.

    If your trained predictor is ready, replace historical_average_flow(...) with:
        predictor.predict_flow(scats_number=u, when=when, model_name="gru")
    """
    graph = graph.copy()
    for u, v, attrs in graph.edges(data=True):
        flow = historical_average_flow(flow_df, u, when)
        attrs["predicted_flow"] = flow
        attrs["travel_time"] = edge_travel_time_minutes(attrs["distance_km"], flow)
    return graph


def top_k_routes(graph: nx.Graph, origin: int, destination: int, k: int = 5) -> list[dict]:
    """Return up to k lowest-time routes."""
    routes = []
    path_generator = nx.shortest_simple_paths(graph, int(origin), int(destination), weight="travel_time")
    for path in path_generator:
        total = sum(graph[path[i]][path[i + 1]]["travel_time"] for i in range(len(path) - 1))
        routes.append({"route": path, "estimated_minutes": round(total, 2)})
        if len(routes) >= k:
            break
    return routes


def route_between(origin: int, destination: int, when_text: str, k: int = 5) -> list[dict]:
    when = pd.to_datetime(when_text).to_pydatetime()
    flow_df = load_time_series(TIME_SERIES_DATA)
    graph = build_knn_graph(CLEANED_DATA)
    weighted_graph = update_edge_weights_with_flows(graph, when, flow_df)
    return top_k_routes(weighted_graph, origin, destination, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--datetime", required=True, help='Example: "2006-10-18 08:15"')
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    for i, route in enumerate(route_between(args.origin, args.destination, args.datetime, args.k), start=1):
        print(f"Route {i}: {route['route']} - {route['estimated_minutes']} minutes")
