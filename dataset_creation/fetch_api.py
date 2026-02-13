from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from annotell.apiclients.EngineApi import EngineApi
from annotell.apiclients.judgement_api_client import JudgementApiClient
from annotell.apiclients.order_execution_api_client import OrderExecutionApiClient
from annotell.apiclients.scene_input_api_client import SceneInputApiClient
from tqdm import tqdm

warnings.filterwarnings("ignore")


class ApiFetcher:
    """API-based fetcher that replaces BigQuery dependency.

    Fetches training data directly from Kognic API using project/request IDs.
    Extracts ALL shapes from judgements (no class filtering).
    Filtering by class name or shape type should be done downstream.
    Caches judgements to disk for faster iteration.
    """

    def __init__(self, config):
        with open(config) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Extract fields from config file
        self.projects = self.config["projects"]
        self.requests = self.config["requests"]
        self.dataset_root = self.config["dataset_root"]
        self.max_workers = self.config.get("max_workers", 8)

        # Determine which ID list to use
        self.id_list = self.requests if self.requests else self.projects
        self.id_list_name = "request_id" if self.requests else "project_id"

        # Initialize API clients
        self.engine_api = EngineApi(env="production")
        self.order_api = OrderExecutionApiClient(env="production")
        self.judgement_api = JudgementApiClient(env="production")
        self.scene_api = SceneInputApiClient(env="production")

        # Setup cache directory
        self.cache_dir = Path(self.dataset_root) / ".cache" / "judgements"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"ApiFetcher initialized for {self.id_list_name}: {self.id_list}")
        print(f"Cache directory: {self.cache_dir}")

    def _get_request_ids(self) -> List[int]:
        """Get request IDs - either from config directly or by looking up a project."""
        if self.requests:
            return list(self.requests)

        request_ids = []
        for project_id in self.projects:
            all_requests = self.engine_api.get_requests()
            project_requests = [req for req in all_requests if req.project_id == project_id]
            print(f"  Project {project_id}: found {len(project_requests)} requests")
            request_ids.extend([req.id for req in project_requests])
        return request_ids

    def _collect_inputs_and_judgements(self) -> Dict[str, Dict]:
        """
        Fetch all inputs and their latest judgement UUIDs.
        Deduplicates by scene UUID (keeps latest judgement if scene appears in multiple requests).

        Returns:
            Dict mapping scene_uuid -> {judgement_uuid, scene_external_id, request_id}
        """
        print(f"\nCollecting inputs and judgements for {self.id_list_name}: {self.id_list}")

        request_ids = self._get_request_ids()
        print(f"Total requests to process: {len(request_ids)}")

        # Collect unique scene -> judgement mapping (dedup by scene_uuid)
        scene_map = {}  # scene_uuid -> {judgement_uuid, scene_external_id, request_id}

        for request_id in tqdm(request_ids, desc="Fetching inputs from requests"):
            try:
                inputs = self.order_api.get_inputs_in_request(request_id)
                for input_data in inputs:
                    scene_uuid = input_data.sceneUuid
                    if not scene_uuid:
                        continue
                    lca = input_data.lastCompletedAnnotation
                    if not lca or not lca.judgementUuid:
                        continue
                    # Keep latest (overwrite if scene already seen - last request wins)
                    scene_map[scene_uuid] = {
                        'judgement_uuid': lca.judgementUuid,
                        'scene_external_id': input_data.sceneExternalId,
                        'request_id': request_id,
                    }
            except Exception as e:
                print(f"  Warning: Failed to get inputs for request {request_id}: {e}")
                continue

        print(f"Unique scenes with completed judgements: {len(scene_map)}")
        return scene_map

    def _resolve_judgement_id(self, judgement_uuid: str) -> Optional[int]:
        """Resolve a judgement UUID to its numeric ID."""
        try:
            meta = self.judgement_api.get_judgement_from_id(judgement_uuid)
            return meta['judgement']['id']
        except Exception:
            return None

    def _fetch_single_judgement(self, judgement_uuid: str) -> Optional[Dict]:
        """
        Fetch a single judgement by UUID, using cache if available.

        Returns:
            Judgement JSON or None if fetch fails
        """
        cache_file = self.cache_dir / f"{judgement_uuid}.json"

        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Warning: Failed to load cached judgement {judgement_uuid}: {e}")

        # Fetch from API (works with both UUID strings and numeric IDs)
        try:
            judgement = self.judgement_api.get_judgement_content(judgement_uuid)

            # Cache to disk
            with open(cache_file, 'w') as f:
                json.dump(judgement, f)

            return judgement

        except Exception as e:
            print(f"  Warning: Failed to fetch judgement {judgement_uuid}: {e}")
            return None

    def _fetch_judgements(self, scene_map: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Fetch all judgements in parallel with caching.
        Also resolves UUIDs to numeric IDs.

        Returns:
            Dict mapping judgement_uuid to {content, numeric_id}
        """
        judgement_uuids = list(set(v['judgement_uuid'] for v in scene_map.values()))
        print(f"\nFetching {len(judgement_uuids)} judgements (using cache where available)...")

        # Fetch judgement content in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            contents = list(tqdm(
                executor.map(self._fetch_single_judgement, judgement_uuids),
                total=len(judgement_uuids),
                desc="Fetching judgements"
            ))

        # Resolve UUIDs to numeric IDs in parallel
        print("Resolving judgement numeric IDs...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            numeric_ids = list(tqdm(
                executor.map(self._resolve_judgement_id, judgement_uuids),
                total=len(judgement_uuids),
                desc="Resolving IDs"
            ))

        # Build dictionary
        judgements = {}
        for judgement_uuid, content, numeric_id in zip(judgement_uuids, contents, numeric_ids):
            if content:
                judgements[judgement_uuid] = {
                    'content': content,
                    'numeric_id': numeric_id,
                }

        print(f"Successfully fetched {len(judgements)} judgements")
        return judgements

    def _extract_shapes(self, judgement: Dict, judgement_id: int) -> List[Dict]:
        """
        Extract all shapes from a judgement.

        Shape-agnostic: works for any shape type (Cube3D, ExtremePointBox, Polygon, etc).
        No class filtering is applied here; filter downstream.

        Args:
            judgement: The judgement JSON
            judgement_id: The judgement ID

        Returns:
            List of shape dicts with structure:
            {
                'shape_id': str,
                'shape_class': str,
                'shape_details': dict (geometry with type, coordinates, rotation, scale, etc.),
                'shape_timestamp': int or None,
                'primarySensor_timestamp': str or None,
                'primarySensor_value': str or None
            }
        """
        shapes = []

        # Get shapes and shapeProperties
        shapes_data = judgement.get("shapes", {})
        shape_properties = judgement.get("shapeProperties", {})

        # Iterate through all sensors/sources
        for sensor_name, sensor_data in shapes_data.items():
            if "features" not in sensor_data:
                continue

            # Iterate through all features (shapes)
            for feature in sensor_data["features"]:
                shape_id = feature.get("id")
                geometry = feature.get("geometry", {})
                properties = feature.get("properties", {})

                if not shape_id or not geometry:
                    continue

                # Get class name from shapeProperties
                shape_props = shape_properties.get(shape_id, {})
                all_props = shape_props.get("@all", {})
                shape_class = all_props.get("class", "")

                # Extract shape type
                shape_type = geometry.get("type", "Unknown")

                # Extract timestamp
                shape_timestamp = properties.get("@timestamp")

                # Match primarySensor timestamp
                primarySensor_timestamp = None
                primarySensor_value = None

                sensor_props = shape_props.get(sensor_name, {})
                primary_sensor_array = sensor_props.get("primarySensor", [])

                if primary_sensor_array and shape_timestamp is not None:
                    # Find matching timestamp
                    for ps_entry in primary_sensor_array:
                        ps_timestamp = ps_entry.get("@timestamp")
                        if ps_timestamp and str(ps_timestamp) == str(shape_timestamp):
                            primarySensor_timestamp = str(ps_timestamp)
                            primarySensor_value = ps_entry.get("value")
                            break

                # Convert numeric arrays in geometry to numpy (matching old pickle format)
                shape_details = dict(geometry)
                for key in ('coordinates', 'rotation', 'scale'):
                    if key in shape_details and isinstance(shape_details[key], list):
                        shape_details[key] = np.array(shape_details[key])

                # Ensure 'unclear' field is present (matching old format)
                if 'unclear' not in shape_details:
                    shape_details['unclear'] = False

                # Build shape dict (matching old pickle format)
                shape_dict = {
                    'shape_id': shape_id,
                    'shape_class': shape_class,
                    'shape_details': shape_details,
                    'shape_timestamp': shape_timestamp,
                    'primarySensor_timestamp': primarySensor_timestamp,
                    'primarySensor_value': primarySensor_value
                }

                shapes.append(shape_dict)

        return shapes

    def _fetch_single_scene_metadata(self, scene_uuid: str) -> Optional[Dict]:
        """
        Fetch scene metadata: timestamp→resource_id mapping and lidar sensor count.

        Returns:
            Dict with 'ts_to_resource' (timestamp→resource_id) and 'sensor_count', or None on failure
        """
        try:
            scene = self.scene_api.get_full_scene(scene_uuid)

            # Build timestamp → resource_id mapping from frames + lidar sensor_frames
            ts_to_resource = {}
            for frame in scene.frames:
                frame_id = frame.frame_id
                rel_ts = frame.relative_timestamp
                # Find resource_id from first lidar resource's sensor_frames
                resource_id = None
                for lr in scene.lidar_resources:
                    sf = lr.sensor_frames.get(str(frame_id))
                    if sf and hasattr(sf, 'resource_id'):
                        resource_id = sf.resource_id
                        break
                if resource_id:
                    ts_to_resource[rel_ts] = resource_id

            # Count lidar sensors from first lidar resource
            sensor_count = 1
            if scene.lidar_resources:
                sensor_count = max(1, len(scene.lidar_resources[0].sensors))

            return {
                'ts_to_resource': ts_to_resource,
                'sensor_count': sensor_count,
            }
        except Exception as e:
            print(f"  Warning: Could not get scene metadata for {scene_uuid}: {e}")
            return None

    def _fetch_scene_metadata(self, scene_uuids: List[str]) -> Dict[str, Dict]:
        """Fetch scene metadata for all scenes in parallel."""
        print(f"\nFetching scene metadata for {len(scene_uuids)} scenes...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self._fetch_single_scene_metadata, scene_uuids),
                total=len(scene_uuids),
                desc="Fetching scene metadata"
            ))

        metadata = {}
        for scene_uuid, result in zip(scene_uuids, results):
            if result:
                metadata[scene_uuid] = result

        print(f"Successfully fetched metadata for {len(metadata)} scenes")
        return metadata

    def get_label_resources(self) -> pd.DataFrame:
        """
        Main entry point: Fetch all training data and return as DataFrame.

        Returns:
            DataFrame with columns:
            - judgement_id
            - input_internal_id (scene UUID)
            - input_n_timestamps
            - resource_id
            - geometries (list of shape dicts)
            - potree_sensor_count
        """
        start_time = time.time()

        # Step 1: Collect inputs and their judgement UUIDs
        scene_map = self._collect_inputs_and_judgements()

        if not scene_map:
            print("No inputs with completed judgements found!")
            return pd.DataFrame()

        # Step 2: Fetch judgement content
        judgements = self._fetch_judgements(scene_map)

        if not judgements:
            print("No judgements fetched!")
            return pd.DataFrame()

        # Step 3: Fetch scene metadata (resource_ids and sensor counts)
        scene_metadata = self._fetch_scene_metadata(list(scene_map.keys()))

        # Step 4: Extract shapes and build DataFrame
        print("\nExtracting shapes...")
        rows = []

        for scene_uuid, info in tqdm(scene_map.items(), desc="Extracting shapes"):
            judgement_uuid = info['judgement_uuid']
            j_data = judgements.get(judgement_uuid)
            if not j_data:
                continue

            content = j_data['content']
            numeric_id = j_data['numeric_id']

            # Extract shapes
            shapes = self._extract_shapes(content, numeric_id or judgement_uuid)

            if not shapes:
                continue

            # Get scene metadata
            s_meta = scene_metadata.get(scene_uuid, {})
            ts_to_resource = s_meta.get('ts_to_resource', {})
            sensor_count = s_meta.get('sensor_count', 1)

            # Group shapes by timestamp (one row per frame, matching old format)
            shapes_by_ts = defaultdict(list)
            for shape in shapes:
                ts = shape.get('shape_timestamp')
                shapes_by_ts[ts].append(shape)

            n_timestamps = len(shapes_by_ts)

            for ts, ts_shapes in shapes_by_ts.items():
                row = {
                    'judgement_id': numeric_id,
                    'input_internal_id': scene_uuid,
                    'input_n_timestamps': n_timestamps,
                    'resource_id': ts_to_resource.get(ts),
                    'geometries': np.array(ts_shapes, dtype=object),
                    'scene_uuid': scene_uuid,
                    'potree_sensor_count': sensor_count,
                }
                rows.append(row)

        # Create DataFrame
        datatable = pd.DataFrame(rows)

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f}s")
        print(f"Total rows: {len(datatable)}")

        # Print stats
        self.print_stats(datatable)

        self.datatable = datatable
        return datatable

    def _build_stats(self, datatable: pd.DataFrame, pickle_path: str = None) -> str:
        """Build statistics string from the fetched data."""
        if len(datatable) == 0:
            return "Empty datatable"

        lines = []
        if pickle_path:
            lines.append(f"Pickle: {pickle_path}")
        lines.append(f"Table shape: {datatable.shape}")
        lines.append(f"Unique scenes: {datatable['input_internal_id'].nunique()}")

        min_seq = datatable["input_n_timestamps"].min()
        max_seq = datatable["input_n_timestamps"].max()
        lines.append(
            f"Frames per sequence: {min_seq}"
            if min_seq == max_seq
            else f"Frames per sequence: {min_seq} - {max_seq}"
        )
        lines.append(f"Total frames: {datatable['input_n_timestamps'].sum()}")

        min_lidar = datatable["potree_sensor_count"].min()
        max_lidar = datatable["potree_sensor_count"].max()
        lines.append(
            f"Lidar sensors: {min_lidar}"
            if min_lidar == max_lidar
            else f"Lidar sensors: {min_lidar} - {max_lidar}"
        )
        lines.append(f"Point clouds: {datatable['potree_sensor_count'].sum()}")

        # Count shapes by class, type, and class+type
        class_counts = Counter()
        type_counts = Counter()
        class_types = defaultdict(Counter)  # class -> {type -> count}
        for geometry in datatable["geometries"]:
            for shape in geometry:
                cls = shape["shape_class"]
                typ = shape["shape_details"].get("type", "Unknown")
                class_counts[cls] += 1
                type_counts[typ] += 1
                class_types[cls][typ] += 1

        lines.append("")
        lines.append("Shape types:")
        for shape_type, count in type_counts.most_common():
            lines.append(f"  {shape_type}: {count}")

        lines.append("")
        lines.append("Shape classes:")
        for shape_class, count in class_counts.most_common():
            types = class_types[shape_class]
            if len(types) == 1:
                typ = next(iter(types))
                lines.append(f"  {shape_class} ({typ}): {count}")
            else:
                lines.append(f"  {shape_class}: {count}")
                for typ, tc in types.most_common():
                    lines.append(f"    {typ}: {tc}")

        return "\n".join(lines)

    def print_stats(self, datatable: pd.DataFrame):
        """Print and persist statistics of the fetched data."""
        stats = self._build_stats(datatable)
        print(f"\nDataset Statistics:\n{stats}")

    def save_stats(self, datatable: pd.DataFrame, config_path: str, pickle_path: str):
        """Save statistics next to the pickle and append to config yaml."""
        stats = self._build_stats(datatable, pickle_path=pickle_path)

        # Write companion stats file next to pickle
        stats_path = pickle_path.replace(".pkl", ".stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats + "\n")
        print(f"Stats written to {stats_path}")

        # Append stats as comments to config yaml (replace previous stats block)
        STATS_MARKER = "# --- Fetch Statistics ---"
        with open(config_path, "r") as f:
            config_content = f.read()

        # Strip any previous stats block
        if STATS_MARKER in config_content:
            config_content = config_content[:config_content.index(STATS_MARKER)].rstrip() + "\n"

        commented_stats = "\n".join(f"# {line}" if line else "#" for line in stats.split("\n"))
        config_content += f"\n{STATS_MARKER}\n{commented_stats}\n"

        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"Stats appended to {config_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fetch_api.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    fetcher = ApiFetcher(config_path)
    datatable = fetcher.get_label_resources()

    # Save to pickle
    output_file = f"datatable_{fetcher.id_list_name}_{','.join(map(str, fetcher.id_list))}_api.pkl"
    datatable.to_pickle(output_file)
    print(f"\nSaved to {output_file}")

    # Save stats
    fetcher.save_stats(datatable, config_path, output_file)
