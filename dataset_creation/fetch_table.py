from collections import Counter
import os
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from google.cloud import bigquery
from tqdm import tqdm

warnings.filterwarnings("ignore", "Unable to determine Arrow type")
warnings.filterwarnings("ignore", "BigQuery Storage module not found")
warnings.filterwarnings("ignore", "Your application has authenticated using")


class FetchTable:
    def __init__(self, config):
        with open(config) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        # Extract fields from config file
        self.table = self.config[
            "table"
        ]  # `annotell-com.dbt_shapes.shapes_training_cuboid`
        self.projects = self.config["projects"]
        self.requests = self.config["requests"]
        self.classes = self.config["classes"]
        if len(self.classes) > 1:
            self.classes = tuple(self.classes)
        else:
            self.classes = f"('{self.classes[0]}')"

        # Determine which ID list to use
        self.id_list = self.requests if self.requests else self.projects
        self.id_list_name = "request_id" if self.requests else "project_id"

    def cuboid_query(self):
        sql_query = f"""
        WITH

        -- Step 1: Get relevant judgement_ids from assignments.db
        relevant_judgement_ids AS (
            SELECT judgement_id
            FROM `annotell-com.dbt_production_metrics.production_metrics_unfiltered`
            where last_exportable_index = sequence_index
        ),

        -- Step 2: Get meta data for relevant judgement_ids, filtering by task_category and project_id
        meta AS (
            SELECT judgement_id, project_id, organization_id, task_category
            FROM annotell-com.dbt_staging__api_data.stg_api_data__judgement_overview
            WHERE judgement_id IN (SELECT judgement_id FROM relevant_judgement_ids)
                AND task_category = 'production'
                AND {self.id_list_name} IN ({", ".join(map(str, self.id_list))})
        ),

        -- Step 3: Get raw data for the filtered judgement_ids
        raw AS (
            SELECT *
            FROM {self.table}
            WHERE judgement_id IN (SELECT judgement_id FROM meta)
        ),

        -- Step 4: Unnest geometries from raw data
        unnested_geometries AS (
            SELECT
                r.judgement_id,
                r.input_internal_id,
                r.input_n_timestamps,
                g.shape_id,
                g.shape_class,
                g.shape_details,
                r.shape_timestamp,
                r.resource_id
            FROM raw r
            CROSS JOIN UNNEST(r.geometries) AS g
        ),

        -- Step 5: Filter geometries by specified shape_classes
        filtered_geometries AS (
            SELECT *
            FROM unnested_geometries
            WHERE shape_class IN {self.classes}
        ),

        -- Step 6: Join filtered geometries with properties to get primarySensor
        geometries_with_properties AS (
            SELECT
                fg.*,
                p.properties_all,
                JSON_EXTRACT(p.properties_all, '$.primarySensor') AS primarySensor_json
            FROM filtered_geometries fg
            LEFT JOIN `dbt_shapes.int_shape_properties` p ON fg.shape_id = p.shape_id
        ),

        -- Step 7: Unnest primarySensor and match timestamps, handling NULLs
        primarySensor_matched AS (
            SELECT
                gwp.judgement_id,
                gwp.input_internal_id,
                gwp.input_n_timestamps,
                gwp.shape_id,
                gwp.shape_class,
                gwp.shape_details,
                gwp.shape_timestamp,
                gwp.resource_id,
                -- Use an ARRAY of primarySensor entries or NULL if primarySensor_json is NULL
                IF(gwp.primarySensor_json IS NOT NULL,
                    ARRAY(
                        SELECT AS STRUCT
                            JSON_VALUE(ps_element, '$.timestamp') AS ps_timestamp,
                            JSON_VALUE(ps_element, '$.value') AS ps_value
                        FROM UNNEST(JSON_QUERY_ARRAY(gwp.primarySensor_json)) AS ps_element
                    ),
                    [STRUCT(
                        CAST(NULL AS STRING) AS ps_timestamp,
                        CAST(NULL AS STRING) AS ps_value
                    )]
                ) AS primarySensor_entries
            FROM geometries_with_properties gwp
        ),

        -- Step 8: Match primarySensor entries to shape_timestamp
        matched_sensors AS (
            SELECT
                psm.judgement_id,
                psm.input_internal_id,
                psm.input_n_timestamps,
                psm.shape_id,
                psm.shape_class,
                psm.shape_details,
                psm.shape_timestamp,
                psm.resource_id,
                -- Find the matching primarySensor entry based on timestamp
                (SELECT AS STRUCT
                    ps_entry.ps_timestamp,
                    ps_entry.ps_value
                FROM UNNEST(psm.primarySensor_entries) AS ps_entry
                WHERE psm.shape_timestamp IS NOT NULL
                AND ps_entry.ps_timestamp = CAST(psm.shape_timestamp AS STRING)
                LIMIT 1
                ) AS matched_primarySensor
            FROM primarySensor_matched psm
        )

        -- Step 9: Prepare final results with deduplicated geometries
        SELECT
            judgement_id,
            input_internal_id,
            input_n_timestamps,
            resource_id,
            ARRAY_AGG(STRUCT(
                shape_id,
                shape_class,
                shape_details,
                shape_timestamp,
                primarySensor_timestamp,
                primarySensor_value
            )) AS geometries
        FROM (
            SELECT
                judgement_id,
                resource_id,
                input_internal_id,
                input_n_timestamps,
                shape_id,
                shape_class,
                ANY_VALUE(shape_details) AS shape_details,
                shape_timestamp,
                matched_primarySensor.ps_timestamp AS primarySensor_timestamp,
                matched_primarySensor.ps_value AS primarySensor_value
            FROM matched_sensors
            GROUP BY
                judgement_id,
                resource_id,
                input_internal_id,
                input_n_timestamps,
                shape_id,
                shape_class,
                shape_timestamp,
                matched_primarySensor.ps_timestamp,
                matched_primarySensor.ps_value
        ) AS deduped_geometries
        GROUP BY
            judgement_id,
            input_internal_id,
            resource_id,
            input_n_timestamps
        """

        print(sql_query)
        return sql_query

    def lidar_sensor_query(self):
        sql_query = f"""
        WITH meta AS (
            SELECT DISTINCT input_internal_id
            FROM annotell-com.dbt_staging__api_data.stg_api_data__judgement_overview
            WHERE {self.id_list_name} IN ({", ".join(map(str, self.id_list))})
            )
        SELECT 
            scene_uuid,
            COUNT(potree_sensor_id) AS potree_sensor_count
        FROM 
            `raw_imports.lidar_sensors` AS lidar
        JOIN 
            meta
        ON 
            lidar.scene_uuid = meta.input_internal_id
        GROUP BY 
            scene_uuid;
        """
        
        return sql_query

    def get_database_table(self, sql_query, desc):
        # Initialize BigQuery client
        client = bigquery.Client()
        job_config = bigquery.QueryJobConfig(use_query_cache=False)
        # Run the query and convert to Pandas DataFrame
        # print("Query:\n", sql_query)
        query_job = client.query(sql_query, job_config=job_config)
        print("query_job:",query_job)
        # Wait a moment to ensure the job starts processing
        time.sleep(1)

        # Monitor the job progress
        with tqdm(
            total=100,
            desc=desc,
        ) as pbar:
            while True:
                query_job.reload()  # Refreshes the state via a GET request.

                if query_job.state == "DONE":
                    if query_job.error_result:
                        print(
                            f"Query failed with error: {query_job.error_result['message']}"
                        )
                        break
                    else:
                        pbar.n = 100
                        pbar.refresh()
                        break
                else:
                    # Estimate progress based on bytes processed
                    total_bytes_processed = int(query_job.total_bytes_processed or 0)
                    total_bytes_estimated = int(
                        query_job.total_bytes_processed or 1
                    )  # Avoid division by zero
                    progress = (total_bytes_processed / total_bytes_estimated) * 100
                    pbar.n = min(int(progress), 99)  # Cap progress at 99% until done
                    pbar.refresh()
                    time.sleep(1)
        # Get the destination table reference
        if query_job.error_result:
            raise RuntimeError(
                f"BigQuery query failed: {query_job.error_result['message']}"
            )
        destination = query_job.destination
        print("destination:", destination)
        destination = client.get_table(destination)  # Fetch the table schema

        # Use list_rows to get a RowIterator
        rows_iter = client.list_rows(destination, page_size=10000)

        # Convert the RowIterator to a DataFrame with a progress bar
        datatable = rows_iter.to_dataframe(progress_bar_type="tqdm")
        return datatable

    def print_stats(self, datatable):
        """
        print statistics of the fetched data
        - number of unique input_internal_id ==> number of unique scenes
        - number of timestamps (min, max if not equal) ==> average number of sequence frames
        - total number of timestamps ==> number of sequence frames
        - number of potree_sensor_count (min, max if not equal) ==> number of lidar sensors
        - sum of potree_sensor_count ==> number of point clouds
        - different shape_class count
        """
        print("Table head:")
        print(datatable.head())
        print(f"Table shape: {datatable.shape}")
        print(f"Number of unique scenes: {datatable['input_internal_id'].nunique()}")
        min_seq_frames = datatable["input_n_timestamps"].min()
        max_seq_frames = datatable["input_n_timestamps"].max()
        print(
            "Frames in sequence:",
            min_seq_frames
            if min_seq_frames == max_seq_frames
            else (min_seq_frames, "-", max_seq_frames),
        )
        print(
            f"Total number of sequence frames: {datatable['input_n_timestamps'].sum()}"
        )
        min_lidar_sensors = datatable["potree_sensor_count"].min()
        max_lidar_sensors = datatable["potree_sensor_count"].max()
        print(
            f"Number of lidar sensors: {min_lidar_sensors}"
            if min_lidar_sensors == max_lidar_sensors
            else f"Number of lidar sensors: {min_lidar_sensors} - {max_lidar_sensors}"
        )
        print(f"Number of point clouds: {datatable['potree_sensor_count'].sum()}")
        # go through geometries and count the number of different shape classes
        classes = []
        for geometry in datatable["geometries"]:
            for shape in geometry:
                classes.append(shape["shape_class"])
        class_counts = Counter(classes)
        df = pd.DataFrame(class_counts.items(), columns=["Class", "Count"])
        print("Different shape classes:")
        # Print the table
        print(df)
        print("First geometry:")
        print(datatable["geometries"].iloc[0][0])

    def get_label_resources(self):
        datatable = self.get_database_table(
            self.cuboid_query(),
            f"""Fetching data table for {f"project {self.config['projects']}" if self.config["projects"] else f"request {self.config['requests']}"}""",
        )
        # print(datatable.head())
        sensor_table = self.get_database_table(
            self.lidar_sensor_query(), """Fetching lidar sensors"""
        )
        # merge the two tables so that input_internal_id in datatable matches scene_uuid in sensor_table
        self.datatable = datatable.merge(
            sensor_table, left_on="input_internal_id", right_on="scene_uuid", how="left"
        )
        return self.datatable


if __name__ == "__main__":
    # fetcher = FetchTable("dataset_creation/config.yaml")
    # fetcher.get_label_resources()
    # load datatable_project_id_178.pkl and print head
    datatable = pd.read_pickle("dataset_creation/datatable_project_id_873,874,915.pkl")
    print(datatable.head())
    # return the scene_uuid for the judgement_id 15952702
    # print(datatable[datatable["judgement_id"] == 15952702]["scene_uuid"].values)
    # print geometries where shape_timestamp is 23012 and judgement_id is 15952702
    subset = datatable[
        (datatable["judgement_id"] == 15952702)
        & (
            datatable["geometries"].apply(
                lambda x: any([i["shape_timestamp"] == 23012 for i in x])
            )
        )
    ]["geometries"]
    for i in subset:
        # print shape_id, shape_details, shape_class, shape_timestamp
        print(
            [
                (
                    j["shape_id"],
                    j["shape_details"],
                    j["shape_class"],
                    j["shape_timestamp"],
                )
                for j in i
            ]
        )
    # find the object where distance to origin of coordinates is the smallest at shape_timestamp 23012
    min_distance = 1000
    min_distance_obj = None
    for geometry in subset:
        for shape in geometry:
            if shape["shape_timestamp"] == 23012:
                distance = np.linalg.norm(
                    np.array(shape["shape_details"]["coordinates"])
                )
                if distance < min_distance:
                    min_distance = distance
                    min_distance_obj = shape
    print(
        f"Object with smallest distance to origin at shape_timestamp 23012: {min_distance_obj}, object distance: {min_distance}"
    )
