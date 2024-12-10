import os
import time
import warnings
import yaml
from google.cloud import bigquery
from tqdm import tqdm

warnings.filterwarnings("ignore", "Unable to determine Arrow type")
warnings.filterwarnings("ignore", "BigQuery Storage module not found")


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
        self.classes_dict = self.config["classes"]
        self.classes = list(self.classes_dict.keys())
        if len(self.classes) > 1:
            self.classes = tuple(self.classes)
        else:
            self.classes = f"('{self.classes[0]}')"

        # Determine which ID list to use
        self.id_list = self.requests if self.requests else self.projects
        self.id_list_name = "request_id" if self.requests else "project_id"

    def create_sql_request(self):
        sql_query = f"""
        WITH

        -- Step 1: Get relevant judgement_ids from assignments.db
        relevant_judgement_ids AS (
            SELECT last_judgement_id_in_chain AS judgement_id
            FROM `annotell-com.dbt_assignment_chains.assignment_chains`
        ),

        -- Step 2: Get meta data for relevant judgement_ids, filtering by task_category and project_id
        meta AS (
            SELECT judgement_id, project_id, organization_id, task_category
            FROM annotell-com.dbt_staging__api_data.stg_api_data__judgement_overview
            WHERE judgement_id IN (SELECT judgement_id FROM relevant_judgement_ids)
                AND task_category = 'production'
                AND {self.id_list_name} IN ({', '.join(map(str, self.id_list))})
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

        return sql_query

    def copy_config(self, cfg):
        dataset_folder = os.path.join(cfg["dataset_root"], cfg["dataset_name"])
        config_path = os.path.join(dataset_folder, "config.yaml")
        with open("config.yaml", "r") as src, open(config_path, "w") as dst:
            dst.write(src.read())

    def get_database_table(self):
        sql_query = self.create_sql_request()
        # Initialize BigQuery client
        client = bigquery.Client()
        job_config = bigquery.QueryJobConfig(use_query_cache=False)
        # Run the query and convert to Pandas DataFrame
        # print("Query:\n", sql_query)
        query_job = client.query(sql_query, job_config=job_config)
        # Wait a moment to ensure the job starts processing
        time.sleep(1)

        # Monitor the job progress
        with tqdm(
            total=100,
            desc=f"""Fetching data table for {f'project {self.config["projects"]}' if self.config["projects"] else f'request {self.config["requests"]}'}""",
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
        destination = query_job.destination
        destination = client.get_table(destination)  # Fetch the table schema

        # Use list_rows to get a RowIterator
        rows_iter = client.list_rows(destination, page_size=10000)

        # Convert the RowIterator to a DataFrame with a progress bar
        self.datatable = rows_iter.to_dataframe(progress_bar_type="tqdm")
        return self.datatable
