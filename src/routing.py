"""Route guidance using predicted travel time."""

from __future__ import annotations

import argparse

import networkx as nx
import pandas as pd

from config import CLEANED_DATA, TIME_SERIES_DATA
from data_processing import get_site_locations, load_time_series
from travel_time import edge_travel_time_minutes, haversine_km


def build_graph(cleaned_csv_path=CLEANED_DATA, neighbours=3):
    """Create a simple graph by connecting each SCATS site to nearby sites."""

    locations = get_site_locations(cleaned_csv_path)
    graph = nx.Graph()

    # Add each SCATS site as a node
    for _, row in locations.iterrows():
        site = int(row["SCATS Number"])

        graph.add_node(
            site,
            label=row["Location"],
            lat=float(row["NB_LATITUDE"]),
            lon=float(row["NB_LONGITUDE"]),
        )

    # Connect each site to its nearest neighbours
    for _, site_row in locations.iterrows():
        site = int(site_row["SCATS Number"])
        lat = float(site_row["NB_LATITUDE"])
        lon = float(site_row["NB_LONGITUDE"])

        distances = []

        for _, other_row in locations.iterrows():
            other_site = int(other_row["SCATS Number"])

            if site == other_site:
                continue

            distance = haversine_km(
                lat,
                lon,
                float(other_row["NB_LATITUDE"]),
                float(other_row["NB_LONGITUDE"]),
            )

            distances.append((distance, other_site))

        nearest_sites = sorted(distances)[:neighbours]

        for distance, other_site in nearest_sites:
            graph.add_edge(
                site,
                other_site,
                distance_km=distance,
                travel_time=0,
            )

    return graph


def get_average_flow(flow_data, scats_number, selected_time):
    """Get average traffic flow for the selected SCATS site and time period."""

    site_data = flow_data[flow_data["SCATS Number"] == int(scats_number)]

    same_time = site_data[
        (site_data["Hour"] == selected_time.hour)
        & (site_data["Minute"] == selected_time.minute)
    ]

    if len(same_time) > 0:
        return float(same_time["Traffic"].mean())

    return float(site_data["Traffic"].mean())


def add_travel_times(graph, selected_time, flow_data):
    """Add travel time to each edge in the graph."""

    graph = graph.copy()

    for start, end, edge_data in graph.edges(data=True):
        flow = get_average_flow(flow_data, start, selected_time)

        travel_time = edge_travel_time_minutes(
            edge_data["distance_km"],
            flow,
        )

        edge_data["predicted_flow"] = flow
        edge_data["travel_time"] = travel_time

    return graph


def find_routes(graph, origin, destination, k=5):
    """Find the top-k shortest routes based on travel time."""

    routes = []

    paths = nx.shortest_simple_paths(
        graph,
        int(origin),
        int(destination),
        weight="travel_time",
    )

    for path in paths:
        total_time = 0

        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            total_time += graph[start][end]["travel_time"]

        routes.append(
            {
                "route": path,
                "estimated_minutes": round(total_time, 2),
            }
        )

        if len(routes) == k:
            break

    return routes


def route_between(origin, destination, datetime_text, k=5):
    """Main function used by main.py to find the routes to the destination."""

    selected_time = pd.to_datetime(datetime_text).to_pydatetime()

    flow_data = load_time_series(TIME_SERIES_DATA)
    graph = build_graph(CLEANED_DATA)
    graph = add_travel_times(graph, selected_time, flow_data)

    return find_routes(graph, origin, destination, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--datetime", required=True)
    parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    routes = route_between(
        args.origin,
        args.destination,
        args.datetime,
        args.k,
    )

    for number, route in enumerate(routes, start=1):
        print(
            f"Route {number}: {route['route']} "
            f"- {route['estimated_minutes']} minutes"
        )