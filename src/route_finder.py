import heapq
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import pandas as pd
from travel_time import haversine_km, edge_travel_time_minutes


class SearchNode:
    def __init__(self, state, parent=None, cost=0, depth=0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.depth = depth
    
    def __lt__(self, other):
        return self.cost < other.cost


def heuristic(graph, site_a: int, site_b: int) -> float:
    if site_a not in graph.nodes or site_b not in graph.nodes:
        return 0.0
    
    lat1 = graph.nodes[site_a]['lat']
    lon1 = graph.nodes[site_a]['lon']
    lat2 = graph.nodes[site_b]['lat']
    lon2 = graph.nodes[site_b]['lon']
    
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    time_minutes = (dist_km / 60.0) * 60 + 0.5  # ~30s per km
    
    return time_minutes


def reconstruct_path(node: SearchNode) -> List[int]:
    path = []
    current = node
    while current is not None:
        path.append(current.state)
        current = current.parent
    return list(reversed(path))


class TrafficGraph:
    def __init__(self, sites_df: pd.DataFrame, distance_matrix: Dict, 
                 predicted_flows: Optional[Dict] = None):
        self.sites_df = sites_df
        self.distance_matrix = distance_matrix
        self.predicted_flows = predicted_flows or {}
        
        # Build nodes dict
        self.nodes = {}
        for _, row in sites_df.iterrows():
            site_id = int(row['SCATS Number'])
            self.nodes[site_id] = {
                'lat': float(row['NB_LATITUDE']),
                'lon': float(row['NB_LONGITUDE']),
                'location': row.get('Location', f'Site {site_id}')
            }
        
        # Build adjacency list
        self.adjacency = self._build_adjacency()
        self.origin = None
        self.destinations = set()
    
    def _build_adjacency(self) -> Dict[int, List[int]]:
        """Connect sites within 2km."""
        adjacency = {site_id: [] for site_id in self.nodes.keys()}
        
        for (site_a, site_b), dist in self.distance_matrix.items():
            if 0 < dist < 2.0:
                if site_b not in adjacency[site_a]:
                    adjacency[site_a].append(site_b)
                if site_a not in adjacency[site_b]:
                    adjacency[site_b].append(site_a)
        
        return adjacency
    
    def set_goal(self, origin: int, destinations):
        self.origin = origin
        self.destinations = set(destinations) if isinstance(destinations, list) else {destinations}
    
    def is_goal(self, node_id: int) -> bool:
        return node_id in self.destinations
    
    def get_neighbors(self, site_id: int) -> List[Tuple[int, float]]:
        """Get neighboring sites and travel times."""
        neighbors = []
        
        for neighbor_id in self.adjacency.get(site_id, []):
            key = (min(site_id, neighbor_id), max(site_id, neighbor_id))
            dist_km = self.distance_matrix.get(key, None)
            
            if dist_km is None:
                continue
            
            flow = self.predicted_flows.get(neighbor_id, 100.0)
            travel_time = edge_travel_time_minutes(dist_km, flow)
            neighbors.append((neighbor_id, travel_time))
        
        return neighbors


def astar(graph: TrafficGraph, origin: int, destination: int, 
          predicted_flows: Optional[Dict] = None) -> Tuple:
    
    if predicted_flows:
        graph.predicted_flows = predicted_flows
    
    graph.set_goal(origin, destination)
    
    visited = set()
    frontier = [SearchNode(state=origin)]

    
    while frontier:
        # Sort by f-score: f = g + h (cost + heuristic)
        frontier.sort(key=lambda x: x.cost + min(
            heuristic(graph, x.state, dest) for dest in graph.destinations
        ))
        
        current = frontier.pop(0)
        
        # Skip if already visited
        if current.state in visited:
            continue
        
        visited.add(current.state)
        
        # Check if goal reached
        if graph.is_goal(current.state):
            return current.state, len(visited), reconstruct_path(current)
        
        # Expand neighbors
        for neighbor_id, travel_cost in graph.get_neighbors(current.state):
            if neighbor_id not in visited:
                child = SearchNode(
                    state=neighbor_id,
                    parent=current,
                    cost=current.cost + travel_cost,
                    depth=current.depth + 1
                )
                frontier.append(child)
    
    # No path found
    return (None, 0, [])


@dataclass
class Route:
    path: List[int]
    total_time: float
    sites_count: int
    
    def __str__(self) -> str:
        path_str = " → ".join(str(s) for s in self.path)
        return f"{path_str} ({self.total_time:.1f} min)"


def astar_k_shortest(graph: TrafficGraph, origin: int, destination: int, 
                     k: int = 5, predicted_flows: Optional[Dict] = None) -> List[Route]:
    """Find k shortest paths using Yen's algorithm."""

    routes = []
    candidates = []
    
    goal_state, visited, first_path = astar(graph, origin, destination, predicted_flows)
    
    if not first_path:
        return []
    
    first_cost = _calculate_path_cost(graph, first_path, predicted_flows)
    routes.append(Route(first_path, first_cost, len(first_path)))
    
    for iteration in range(k - 1):
        if not routes:
            break
        
        last_path = routes[-1].path
        
        for j in range(len(last_path) - 1):
            root_path = last_path[:j + 1]
            spur_node = last_path[j]
            
            removed_edges = []
            for p in routes:
                if len(p.path) > j + 1 and p.path[:j + 1] == root_path:
                    prev_node = p.path[j]
                    next_node = p.path[j + 1]
                    
                    if next_node in graph.adjacency.get(prev_node, []):
                        graph.adjacency[prev_node].remove(next_node)
                        removed_edges.append((prev_node, next_node))
            
            goal_state, _, spur_path = astar(graph, spur_node, destination, predicted_flows)
            
            for prev_node, next_node in removed_edges:
                if next_node not in graph.adjacency[prev_node]:
                    graph.adjacency[prev_node].append(next_node)
            
            if spur_path:
                candidate_path = root_path[:-1] + spur_path
                candidate_cost = _calculate_path_cost(graph, candidate_path, predicted_flows)
                
                is_unique = not any(p.path == candidate_path for p in routes + 
                                  [Route(c[1], c[0], len(c[1])) for c in candidates])
                
                if is_unique:
                    candidates.append((candidate_cost, candidate_path))
        
        if not candidates:
            break
        
        candidates.sort(key=lambda x: x[0])
        best_cost, best_path = candidates.pop(0)
        routes.append(Route(best_path, best_cost, len(best_path)))
    
    return sorted(routes[:k], key=lambda x: x.total_time)


def _calculate_path_cost(graph: TrafficGraph, path: List[int], 
                         predicted_flows: Optional[Dict] = None) -> float:
    if predicted_flows:
        graph.predicted_flows = predicted_flows
    
    total_cost = 0.0
    for i in range(len(path) - 1):
        neighbors = graph.get_neighbors(path[i])
        for neighbor_id, travel_time in neighbors:
            if neighbor_id == path[i + 1]:
                total_cost += travel_time
                break
    
    return total_cost


def build_distance_matrix(sites_df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    distance_matrix = {}
    sites_list = sites_df['SCATS Number'].astype(int).unique()
    
    for i, site_a in enumerate(sites_list):
        row_a = sites_df[sites_df['SCATS Number'] == site_a].iloc[0]
        
        for site_b in sites_list[i + 1:]:
            row_b = sites_df[sites_df['SCATS Number'] == site_b].iloc[0]
            
            lat_a = float(row_a['NB_LATITUDE'])
            lon_a = float(row_a['NB_LONGITUDE'])
            lat_b = float(row_b['NB_LATITUDE'])
            lon_b = float(row_b['NB_LONGITUDE'])
            
            dist = haversine_km(lat_a, lon_a, lat_b, lon_b)
            distance_matrix[(site_a, site_b)] = dist
    
    return distance_matrix


def load_boroondara_graph(sites_csv: str, 
                          predicted_flows: Optional[Dict] = None) -> TrafficGraph:
    sites_df = pd.read_csv(sites_csv)
    distance_matrix = build_distance_matrix(sites_df)
    return TrafficGraph(sites_df, distance_matrix, predicted_flows)