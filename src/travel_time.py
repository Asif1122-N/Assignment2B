"""Convert predicted traffic flow into edge travel time."""
from __future__ import annotations

import math

from config import INTERSECTION_DELAY_SECONDS, SPEED_LIMIT_KMH


def flow_to_speed_kmh(flow: float, assume_under_capacity: bool = True) -> float:
    """Convert vehicles/hour flow to estimated speed using the given quadratic.

    Assignment formula:
        flow = -1.4648375 * speed^2 + 93.75 * speed

    For this assignment we normally assume traffic is under capacity, so we use
    the higher-speed branch and cap it at the speed limit. If congested mode is
    requested, the lower-speed branch is selected instead.
    """
    flow = max(float(flow), 0.0)
    a = -1.4648375
    b = 93.75
    c = -flow
    discriminant = max(b * b - 4 * a * c, 0.0)
    root1 = (-b + math.sqrt(discriminant)) / (2 * a)
    root2 = (-b - math.sqrt(discriminant)) / (2 * a)
    roots = [r for r in (root1, root2) if r > 0]
    if not roots:
        return 5.0

    speed = max(roots) if assume_under_capacity else min(roots)
    speed = min(speed, SPEED_LIMIT_KMH)
    return max(speed, 5.0) 

def edge_travel_time_minutes(distance_km: float, predicted_flow_per_hour: float) -> float:
    """Estimate travel time for one road segment in minutes."""
    speed = flow_to_speed_kmh(predicted_flow_per_hour)
    driving_minutes = (float(distance_km) / speed) * 60.0
    delay_minutes = INTERSECTION_DELAY_SECONDS / 60.0
    return driving_minutes + delay_minutes


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two latitude/longitude points in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))
