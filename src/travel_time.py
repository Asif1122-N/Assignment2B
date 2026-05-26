"""Convert predicted traffic flow into the time travelled."""

from __future__ import annotations

import math

from config import INTERSECTION_DELAY_SECONDS, SPEED_LIMIT_KMH


def flow_to_speed_kmh(flow: float) -> float:
    """Convert predicted traffic state into speed."""

    flow = max(float(flow), 0)

    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return 5.0

    speed_1 = (-b + math.sqrt(discriminant)) / (2 * a)
    speed_2 = (-b - math.sqrt(discriminant)) / (2 * a)

    speed = max(speed_1, speed_2)

    if speed > SPEED_LIMIT_KMH:
        speed = SPEED_LIMIT_KMH

    if speed < 5:
        speed = 5.0

    return speed


def edge_travel_time_minutes(distance_km: float, predicted_flow_per_hour: float) -> float:
    """Calculate travel time for one road."""

    speed = flow_to_speed_kmh(predicted_flow_per_hour)

    driving_time = (distance_km / speed) * 60
    intersection_delay = INTERSECTION_DELAY_SECONDS / 60

    return driving_time + intersection_delay


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points of the map."""

    earth_radius = 6371.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1

    a = (
        math.sin(lat_diff / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(lon_diff / 2) ** 2
    )

    distance = 2 * earth_radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return distance