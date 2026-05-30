"""We want to find the best route between the origin and destination SCATS sites at a given time.
To do this, we will:
1. Build a graph of the road network using the SCATS site locations as nodes and edges connecting nearby sites.
2. Use XGBoost to predict traffic flow at the destination site for a given time, convert using the formula to travel time, and set this as the edge weight.
3. Use K-shortest path algorithm to find the best routes based on the predicted travel times.
"""

import math
import argparse

import joblib
import networkx as nx
import pandas as pd

from config import CLEANED_DATA, MODEL_DIR
from data_processing import get_site_locations
from travel_time import edge_travel_time_minutes, haversine_km


# Load the trained XGBoost model when this file is imported.
_bundle = joblib.load(MODEL_DIR / "xgboost.joblib")
_model = _bundle["model"]
_feature_cols = _bundle["feature_cols"]


def predict_flow(scats_number, selected_time):
    """Predicting how many cars are passing the site at a given time."""

    hour = selected_time.hour
    minute = selected_time.minute
    day = selected_time.weekday()
    slot = hour * 4 + minute // 15

    # Establishing the default values for the lag and rolling mean features. 
    row = pd.DataFrame([{
        "SCATS Number": scats_number,
        "Hour": hour,
        "Minute": minute,
        "DayOfWeek": day,
        "IsWeekend": int(day >= 5),
        "TimeSin": math.sin(2 * math.pi * slot / 96),
        "TimeCos": math.cos(2 * math.pi * slot / 96),
        "DaySin": math.sin(2 * math.pi * day / 7),
        "DayCos": math.cos(2 * math.pi * day / 7),
        "Lag1": 80.0,
        "Lag2": 80.0,
        "Lag4": 75.0,
        "Lag8": 75.0,
        "Lag96": 70.0,
        "RollingMean4": 78.0,
        "RollingMean8": 76.0,
    }])

    prediction = _model.predict(row[_feature_cols])

    # This ensure taht the flow can never be negative, which can bring errors in time travel calculation.
    return max(float(prediction[0]), 0.0)


def build_graph(neighbours=5):
    """Build a road network graph from the SCATS site locations."""

    locations = get_site_locations(CLEANED_DATA)
    graph = nx.Graph()

    # Add every SCATS site with its latitude and longitude as a node in the graph
    for _, row in locations.iterrows():
        site = int(row["SCATS Number"])
        graph.add_node(
            site,
            label=row["Location"],
            lat=float(row["NB_LATITUDE"]),
            lon=float(row["NB_LONGITUDE"]),
        )

    # Connect SITES to the better alternative (neighbours) based on distance
    for _, site_row in locations.iterrows():
        site = int(site_row["SCATS Number"])
        lat = float(site_row["NB_LATITUDE"])
        lon = float(site_row["NB_LONGITUDE"])

        # Calculate distance to every other site
        distances = []
        for _, other_row in locations.iterrows():
            other = int(other_row["SCATS Number"])
            if other == site:
                continue
            dist = haversine_km(lat, lon,
                                float(other_row["NB_LATITUDE"]),
                                float(other_row["NB_LONGITUDE"]))
            distances.append((dist, other))

      
        for dist, other in sorted(distances)[:neighbours]:
            graph.add_edge(site, other, distance_km=dist, travel_time=0)

    return graph


def add_travel_times(graph, selected_time):
    """Set the travel time on every edge using the ML predicted flow."""

    graph = graph.copy()

    for start, end, data in graph.edges(data=True):
        # Predict how busy the intersection at the end of this edge will be at the selected time
        flow = predict_flow(end, selected_time)

        # Convert teh predicted flow into a travel time for this edge using the formula from the assignment description
        time = edge_travel_time_minutes(data["distance_km"], flow)

        data["predicted_flow"] = flow
        data["travel_time"] = time

    return graph


def find_routes(graph, origin, destination, k=5):
    """Return up to k routes sorted by estimated travel time."""

    routes = []

    for path in nx.shortest_simple_paths(graph, int(origin), int(destination), weight="travel_time"):
        # Add up the travel time along this path
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


def route_between(origin, destination, datetime_text, k=5):
    """The main routing function."""

    selected_time = pd.to_datetime(datetime_text).to_pydatetime()

    graph = build_graph()
    graph = add_travel_times(graph, selected_time)

    return find_routes(graph, origin, destination, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--destination", type=int, required=True)
    parser.add_argument("--datetime", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    for number, route in enumerate(route_between(args.origin, args.destination, args.datetime, args.k), start=1):
        print(f"Route {number}: {route['route']} - {route['estimated_minutes']} minutes")
