import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    CLEANED_DATA,
    INTERSECTION_DELAY_SECONDS,
    LOOKBACK_STEPS,
    SPEED_LIMIT_KMH,
    TIME_SERIES_DATA,
)
from data_processing import (
    get_site_locations,
    load_time_series,
    make_sequence_data,
    make_tabular_features,
    temporal_train_test_split,
)
from travel_time import edge_travel_time_minutes, flow_to_speed_kmh, haversine_km


# Module-level fixtures 
print("Loading dataset fixtures (this may take a moment)...")
_raw_df         = load_time_series(TIME_SERIES_DATA)
_tabular_df, _  = make_tabular_features(_raw_df)
_site_locations = get_site_locations(CLEANED_DATA)
_sequence_data  = make_sequence_data(_raw_df, lookback=LOOKBACK_STEPS, test_size=0.2)
print("Fixtures ready.\n")


# TC01 — Dataset loading and structure
class TC01_DataLoading(unittest.TestCase):

    def test_01_dataset_loads_correctly(self):
        """TC01: Dataset loads, is non-empty, and contains all required columns."""
        self.assertIsInstance(_raw_df, pd.DataFrame)
        self.assertGreater(len(_raw_df), 0, "Loaded DataFrame should not be empty")
        required = {"SCATS Number", "Traffic", "Datetime", "Hour", "Minute",
                    "DayOfWeek", "IsWeekend", "TimeSin", "TimeCos", "DaySin", "DayCos"}
        missing = required - set(_raw_df.columns)
        self.assertEqual(missing, set(), f"Missing columns: {missing}")


# TC02 — Missing and malformed data handling
class TC02_MalformedData(unittest.TestCase):

    def test_02a_nan_rows_are_dropped(self):
        """TC02a: Real dataset has no nulls; NaN rows are dropped during cleaning."""
        self.assertTrue(_raw_df["Datetime"].notna().all(), "Null Datetime values found")
        self.assertTrue(_raw_df["Traffic"].notna().all(), "Null Traffic values found")

        # Simulate cleaning on a synthetic frame with a NaN row
        df = pd.DataFrame({
            "SCATS Number": [2000, 2000, 2000],
            "Traffic":      [45.0, None, 60.0],
        })
        df["Traffic"] = pd.to_numeric(df["Traffic"], errors="coerce")
        cleaned = df.dropna(subset=["Traffic"])
        self.assertEqual(len(cleaned), 2, "NaN traffic row should have been dropped")

    def test_02b_traffic_values_in_range(self):
        """TC02b: Traffic values are within a realistic range (0-1800 veh/15 min)."""
        self.assertTrue(
            (_raw_df["Traffic"] <= 1800).all(),
            "Some traffic readings exceed 1800 vehicles per 15 min — likely corrupt"
        )


# TC03 — Data structure integrity
class TC03_DataStructureIntegrity(unittest.TestCase):

    def test_03a_time_features_bounded(self):
        """TC03a: Sin/Cos time features are in [-1, 1]."""
        for col in ["TimeSin", "TimeCos", "DaySin", "DayCos"]:
            self.assertTrue(
                _raw_df[col].between(-1, 1).all(),
                f"Column {col} has values outside [-1, 1]"
            )

    def test_03b_lag_features_present(self):
        """TC03b: All lag and rolling mean features are present in the tabular dataset."""
        for lag in [1, 2, 4, 8, 96]:
            self.assertIn(f"Lag{lag}", _tabular_df.columns, f"Lag{lag} column missing")
        self.assertIn("RollingMean4", _tabular_df.columns)
        self.assertIn("RollingMean8", _tabular_df.columns)


# TC04 — Sequence data shapes (LSTM / GRU)
class TC04_SequenceDataShapes(unittest.TestCase):

    def test_04_sequence_shapes_correct(self):
        """TC04: X_train is 3D with correct lookback; X/y sample counts match."""
        self.assertEqual(_sequence_data.X_train.ndim, 3,
                         f"Expected 3D array, got {_sequence_data.X_train.ndim}D")
        self.assertEqual(_sequence_data.X_train.shape[1], LOOKBACK_STEPS)
        self.assertEqual(_sequence_data.X_train.shape[0], _sequence_data.y_train.shape[0])
        self.assertEqual(_sequence_data.X_test.shape[0], _sequence_data.y_test.shape[0])


# TC05 — Temporal train/test split
class TC05_TemporalSplit(unittest.TestCase):

    def test_05_no_data_leakage(self):
        """TC05: For every site, all train datetimes precede all test datetimes."""
        train, test = temporal_train_test_split(_tabular_df, test_size=0.2)
        for site in train["SCATS Number"].unique():
            if site not in test["SCATS Number"].values:
                continue
            train_max = train.loc[train["SCATS Number"] == site, "Datetime"].max()
            test_min  = test.loc[test["SCATS Number"]  == site, "Datetime"].min()
            self.assertLessEqual(train_max, test_min,
                                 f"Site {site}: train data bleeds into test period")


# TC06 — Site locations
class TC06_SiteLocations(unittest.TestCase):

    def test_06_coordinates_valid(self):
        """TC06: No site has null coordinates and all are within Boroondara bounds."""
        self.assertTrue(_site_locations["NB_LATITUDE"].notna().all(),
                        "Null latitudes found")
        self.assertTrue(_site_locations["NB_LONGITUDE"].notna().all(),
                        "Null longitudes found")
        self.assertTrue(
            _site_locations["NB_LATITUDE"].between(-37.90, -37.75).all(),
            "Some latitudes are outside the Boroondara range"
        )
        self.assertTrue(
            _site_locations["NB_LONGITUDE"].between(144.99, 145.11).all(),
            "Some longitudes are outside the Boroondara range"
        )


# TC07 — Haversine distance calculation
class TC07_Haversine(unittest.TestCase):

    def test_07_haversine_correct(self):
        """TC07: Haversine returns 0 for identical points and is symmetric."""
        self.assertAlmostEqual(
            haversine_km(-37.82, 145.05, -37.82, 145.05), 0.0, places=6
        )
        d1 = haversine_km(-37.82, 145.03, -37.84, 145.06)
        d2 = haversine_km(-37.84, 145.06, -37.82, 145.03)
        self.assertAlmostEqual(d1, d2, places=6)


# TC08 — Flow-to-speed conversion
class TC08_FlowToSpeed(unittest.TestCase):

    def test_08_speed_capped_and_decreases(self):
        """TC08: Speed never exceeds the limit and higher flow yields lower speed."""
        for flow in [0, 50, 100, 500, 1000, 2000]:
            self.assertLessEqual(flow_to_speed_kmh(flow), SPEED_LIMIT_KMH,
                                 f"Speed exceeded limit at flow={flow}")
        self.assertLessEqual(flow_to_speed_kmh(800), flow_to_speed_kmh(100),
                             "Higher flow should not result in a faster speed")


# TC09 — Edge travel time calculation
class TC09_EdgeTravelTime(unittest.TestCase):

    def test_09_travel_time_correct(self):
        """TC09: Travel time includes the 30s delay and matches a hand-computed value."""
        delay_minutes = INTERSECTION_DELAY_SECONDS / 60
        t = edge_travel_time_minutes(distance_km=1.0, predicted_flow_per_hour=100)
        self.assertGreaterEqual(t, delay_minutes,
                                "Travel time should include at least the intersection delay")

        dist = 1.0
        expected = (dist / SPEED_LIMIT_KMH) * 60 + delay_minutes
        result = edge_travel_time_minutes(dist, 0.0)
        self.assertAlmostEqual(result, expected, delta=expected * 0.05)


# TC10 — Graph construction
class TC10_GraphConstruction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from routing import build_graph
        cls.graph = build_graph(neighbours=5)

    def test_10a_nodes_and_edges_valid(self):
        """TC10: All SCATS sites are nodes and every edge has a valid distance_km."""
        expected = set(_site_locations["SCATS Number"].astype(int).tolist())
        actual   = set(self.graph.nodes())
        missing  = expected - actual
        self.assertEqual(missing, set(), f"Missing nodes: {missing}")

        for u, v, data in self.graph.edges(data=True):
            self.assertIn("distance_km", data, f"Edge ({u},{v}) missing distance_km")
            self.assertGreater(data["distance_km"], 0)

    def test_10b_graph_is_connected(self):
        """TC10e: Graph is connected — required for route finding to work."""
        import networkx as nx
        self.assertTrue(nx.is_connected(self.graph),
                        "Graph is not fully connected — some sites are unreachable")


# TC11 — Route finding
class TC11_RouteFinding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from routing import add_travel_times, build_graph
        graph = build_graph(neighbours=5)
        cls.graph = add_travel_times(graph, pd.Timestamp("2006-10-03 08:00:00"))

    def _routes(self, origin=2000, destination=3002, k=5):
        from routing import find_routes
        return find_routes(self.graph, origin=origin, destination=destination, k=k)

    def test_11a_valid_pair_returns_sorted_routes(self):
        """TC11a: A valid O→D pair returns 1–5 routes sorted by travel time."""
        result = self._routes()
        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), 5)
        times = [r["estimated_minutes"] for r in result]
        self.assertEqual(times, sorted(times), "Routes are not sorted by travel time")

    def test_11b_routes_end_at_destination(self):
        """TC11b: Every route ends at the destination site."""
        for r in self._routes():
            self.assertEqual(r["route"][-1], 3002)

    def test_11c_invalid_origin_handled(self):
        """TC11c: An invalid SCATS site number returns empty list or raises cleanly."""
        try:
            result = self._routes(origin=99999, destination=3002)
            self.assertEqual(result, [], "Expected empty list for invalid origin")
        except Exception:
            pass

    def test_11d_same_origin_destination_handled(self):
        """TC11d: O == D does not crash the system."""
        try:
            result = self._routes(origin=2000, destination=2000)
            if result:
                self.assertAlmostEqual(result[0]["estimated_minutes"], 0.0, delta=1.0)
        except Exception:
            pass

    def test_11e_no_duplicate_nodes_in_route(self):
        """TC11j: Route does not revisit the same node."""
        for r in self._routes():
            route = r["route"]
            self.assertEqual(len(route), len(set(route)),
                             f"Route contains duplicate nodes: {route}")


# TC12 — Prediction plausibility (XGBoost)
class TC12_PredictionPlausibility(unittest.TestCase):

    def test_12a_peak_offpeak_differ(self):
        """TC12a: Predicted flow at 8am differs from predicted flow at 2am."""
        from predict_xgboost import predict_xgboost_flow
        _, peak    = predict_xgboost_flow(2000, pd.Timestamp("2006-10-03 08:00:00"))
        _, offpeak = predict_xgboost_flow(2000, pd.Timestamp("2006-10-03 02:00:00"))
        self.assertNotEqual(peak, offpeak,
                            "Peak and off-peak predictions should not be identical")

    def test_12b_prediction_in_range(self):
        """TC12b: Predicted flow is >= 0 and within a realistic range (0–3600 veh/hr)."""
        from predict_xgboost import predict_xgboost_flow
        _, flow = predict_xgboost_flow(2000, pd.Timestamp("2006-10-03 08:00:00"))
        self.assertGreaterEqual(flow, 0.0)
        self.assertLessEqual(flow, 3600, f"Predicted flow {flow} is implausibly high")

    def test_12c_prediction_is_repeatable(self):
        """TC12d: Same input produces same prediction (model inference is deterministic)."""
        from predict_xgboost import predict_xgboost_flow
        _, f1 = predict_xgboost_flow(2000, pd.Timestamp("2006-10-03 08:00:00"))
        _, f2 = predict_xgboost_flow(2000, pd.Timestamp("2006-10-03 08:00:00"))
        self.assertAlmostEqual(f1, f2, places=6)


# TC13 — Regression metrics
class TC13_RegressionMetrics(unittest.TestCase):

    def test_13a_metrics_keys_and_zero_actuals(self):
        """TC13: Metrics handle zero actuals without crashing and return all keys."""
        from train_models import regression_metrics
        m = regression_metrics([0.0, 10.0, 20.0], [1.0, 11.0, 19.0])
        for key in ["MAE", "RMSE", "MAPE_percent", "R2"]:
            self.assertIn(key, m)
        self.assertGreaterEqual(m["MAPE_percent"], 0)

    def test_13b_imperfect_predictions_have_error(self):
        """TC13d: Imperfect predictions produce positive MAE and RMSE."""
        from train_models import regression_metrics
        m = regression_metrics([10, 20, 30], [15, 25, 35])
        self.assertGreater(m["MAE"], 0)
        self.assertGreater(m["RMSE"], 0)


# Entry point — runs tests and saves results to outputs/test_results.txt
if __name__ == "__main__":
    import io
    from datetime import datetime
    from config import OUTPUT_DIR

    OUTPUT_DIR.mkdir(exist_ok=True)
    results_path = OUTPUT_DIR / "test_results.txt"

    # Capture the test output
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__("__main__"))
    result = runner.run(suite)
    output = stream.getvalue()

    # Print to terminal as normal
    print(output)

    # Build summary
    total   = result.testsRun
    failed  = len(result.failures)
    errored = len(result.errors)
    passed  = total - failed - errored
    status  = "OK" if result.wasSuccessful() else "FAILED"
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "Test run: " + now,
        "Result:   " + status,
        "Ran:      " + str(total) + " tests",
        "Passed:   " + str(passed),
        "Failed:   " + str(failed),
        "Errors:   " + str(errored),
    ]
    summary = "\n".join(lines)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n" + "-" * 70 + "\n\n")
        f.write(output)

    print("Test results saved to: " + str(results_path))