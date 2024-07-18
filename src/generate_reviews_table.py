#!/usr/bin/env python
# coding: utf-8

import argparse
import gc

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
    Main function to fetch aggregated product reviews from BigQuery and save to MongoDB.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    args = parser.parse_args()

    object_name = "reviews_product"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:

        client = bigquery.Client()

        query = f"SELECT * FROM ww-da-ingestion.v_extract1.pdt_reviews_aggregated"

        df = client.query(query).to_dataframe()

        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=df, object_name=object_name)


if __name__ == "__main__":
    main()
