#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
from functools import reduce

import pandas as pd
import yaml
from tqdm import tqdm

from bigquery_handlers import BigQueryDataProcessor
from mongodb_lib import *

# Load configuration from YAML files
config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
bigquery_config = config["bigquery-to-retrieve"]
key_field = bigquery_config["key-field"]
tables = bigquery_config["tables"]

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def merge_dfs(left, right):
    """
    Merge two DataFrames on key_field with an outer join.

    Parameters:
    left (pd.DataFrame): The left DataFrame to merge.
    right (pd.DataFrame): The right DataFrame to merge.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    return pd.merge(left, right, on=key_field, how="outer")


def main():
    """
    This script retrieves various datasets from a BigQuery database, merges them
    into a single DataFrame, and saves the resulting DataFrame as a pickle file.

    Steps:
    1. Read multiple tables from BigQuery into DataFrames.
    2. Merge the DataFrames into a single DataFrame using a common key field.
    3. Save the merged DataFrame as a pickle file if it doesn't already exist.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )

    args = parser.parse_args()

    object_name = "product_tables"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:
        # Initialize an empty list to store DataFrames
        dfs = []

        # Iterate over each table specified in the configuration
        for bigquery_table in tqdm(tables):
            # Retrieve table elements from the configuration
            table_elements = tables[bigquery_table]

            # Determine if time feature is required based on the table
            time_feature = bigquery_table == "bookings"

            # Create a BigQueryDataProcessor instance to handle data retrieval
            bqdp = BigQueryDataProcessor(
                config=config,
                dataset_id=table_elements["dataset_id"],
                table_id=bigquery_table,
                table_fields=table_elements["fields"],
                key_field=key_field,
                time_feature=time_feature,
            )
            # Read the BigQuery table into a DataFrame
            bqdp.read_bigquery_table()

            # Append the retrieved DataFrame to the list
            dfs.append(bqdp.df)

        # Merge all DataFrames in the list into a single DataFrame
        product_df = reduce(merge_dfs, dfs)

        # Save the merged DataFrame as a pickle file
        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=product_df, object_name=object_name)

    else:
        print("Skipping retrieving BigQuery data.")


if __name__ == "__main__":
    main()
