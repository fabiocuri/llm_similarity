#!/usr/bin/env python
# coding: utf-8

import argparse
import gc

import numpy as np
import yaml
from google.cloud import bigquery

from mongodb_lib import *

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def main():
    """
    Main function to fetch latest product prices from BigQuery and save to MongoDB.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    args = parser.parse_args()

    object_name = "price_product_tourgrade"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:
        client = bigquery.Client()

        # Define your SQL query with ROW_NUMBER() to get the latest SeasonTo for each ProductCode
        query = """
            WITH ranked_data AS (
                SELECT
                    ProductCode,
                    TourGradeCode,
                    SeasonTo,
                    adult_retail_price,
                    ROW_NUMBER() OVER (PARTITION BY ProductCode, TourGradeCode ORDER BY SeasonTo DESC) AS row_num
                FROM
                    `v_extract1.product_price_time_series`
            )
            SELECT
                ProductCode,
                TourGradeCode,
                SeasonTo,
                adult_retail_price
            FROM
                ranked_data
            WHERE
                row_num = 1
        """

        # Run the query and retrieve the data into a DataFrame
        df = client.query(query).to_dataframe()
        df["adult_retail_price"] = df["adult_retail_price"].replace("", np.nan)
        df["adult_retail_price"] = df["adult_retail_price"].astype(float)

        product_stats = (
            df.groupby(["ProductCode", "TourGradeCode"])["adult_retail_price"]
            .agg(["mean"])
            .reset_index()
        )

        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=product_stats, object_name=object_name)


if __name__ == "__main__":
    main()
