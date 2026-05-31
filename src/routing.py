"""We want to find the best route between the origin and destination SCATS sites at a given time.
To do this, we will:
1. Build a graph of the road network using the SCATS site locations as nodes and edges connecting nearby sites.
2. Use XGBoost to predict traffic flow at the destination site for a given time, convert using the formula to travel time, and set this as the edge weight.
3. Use K-shortest path algorithm to find the best routes based on the predicted travel times.
"""

from __future__ import annotations

import argparse

import networkx as nx
import pandas as pd

from config import CLEANED_DATA
from data_processing import get_site_locations
from predict_xgboost import predict_xgboost_flow
from travel_time import edge_travel_time_minutes, haversine_km


# 1. Predict traffic flow

def predict_flow(scats_number, selected_time):
    """Predict traffic flow using the real XGBoost prediction function."""

    predicted_15_min, predicted_per_hour = predict_xgboost_flow(
        scats_number=scats_number,
        datetime_value=selected_time,
    )

    return max(float(predicted_per_hour), 0.0)


# 2. Build graph

def build_graph(neighbours=5):
    """Build a road network graph from SCATS site locations."""

    locations = get_site_locations(CLEANED_DATA)
    graph = nx.Graph()

    for _, row in locations.iterrows():
        site = int(row["SCATS Number"])

        graph.add_node(
            site,
            label=row["Location"],
            lat=float(row["NB_LATITUDE"]),
            lon=float(row["NB_LONGITUDE"]),
        )

    for _, site_row in locations.iterrows():
        site = int(site_row["SCATS Number"])
        lat = float(site_row["NB_LATITUDE"])
        lon = float(site_row["NB_LONGITUDE"])

        distances = []

        for _, other_row in locations.iterrows():
            other = int(other_row["SCATS Number"])

            if other == site:
                continue

            dist = haversine_km(
                lat,
                lon,
                float(other_row["NB_LATITUDE"]),
                float(other_row["NB_LONGITUDE"]),
            )

            distances.append((dist, other))

        for dist, other in sorted(distances)[:neighbours]:
            graph.add_edge(
                site,
                other,
                distance_km=dist,
                travel_time=0,
            )

    return graph


# 3. Add travel times to graph edges

def add_travel_times(graph, selected_time):
    """Set travel time on every edge using XGBoost predicted flow."""

    graph = graph.copy()

    for start, end, data in graph.edges(data=True):
        flow = predict_flow(
            scats_number=end,
            selected_time=selected_time,
        )

        time = edge_travel_time_minutes(
            distance_km=data["distance_km"],
            predicted_flow_per_hour=flow,
        )

        data["predicted_flow"] = flow
        data["travel_time"] = time

    return graph


# 4. Find top-k routes

def find_routes(graph, origin, destination, k=5):
    """Return up to k routes sorted by estimated travel time."""

    routes = []

    for path in nx.shortest_simple_paths(
        graph,
        int(origin),
        int(destination),
        weight="travel_time",
    ):
        total_time = 0

        for i in range(len(path) - 1):
            total_time += graph[path[i]][path[i + 1]]["travel_time"]

        routes.append({
            "route": path,
            "estimated_minutes": round(total_time, 2),
        })

        if len(routes) == k:
            break

    return routes


# 5. Main route function

def route_between(origin, destination, datetime_text, k=5):
    """Find best routes between origin and destination."""

    selected_time = pd.to_datetime(datetime_text)

    graph = build_graph()
    graph = add_travel_times(graph, selected_time)

    return find_routes(
        graph=graph,
        origin=origin,
        destination=destination,
        k=k,
    )


# 6. Command line test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--datetime", required=True)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    routes = route_between(
        origin=args.origin,
        destination=args.destination,
        datetime_text=args.datetime,
        k=args.k,
    )

    for number, route in enumerate(routes, start=1):
        print(
            f"Route {number}: {route['route']} "
            f"- {route['estimated_minutes']} minutes"
        )