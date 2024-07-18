#!/usr/bin/env python
# coding: utf-8

import argparse
import gc

import yaml
from ydata_profiling import ProfileReport

from mongodb_lib import *
from preprocessing_handlers import DataFrameProcessor

# Load configuration from YAML files.
config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
bigquery_config = config["bigquery-to-retrieve"]
key_field = bigquery_config["key-field"]
location_field = bigquery_config["location-field"]

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def main():
    """
    Main function to process product data and generate profiling reports.

    Steps:
    1. Initialize a DataFrameProcessor with the product data.
    2. Preprocess the data.
    3. Save the processed tabular and textual data as pickle files.
    4. Generate and save profiling reports for both tabular and textual data.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )

    args = parser.parse_args()

    tabular_object_name = "product_tabular"
    tabular_existing_file = fs.find_one({"filename": tabular_object_name})

    textual_object_name = "product_textual"
    textual_existing_file = fs.find_one({"filename": textual_object_name})

    if not tabular_existing_file or not textual_existing_file or args.overwrite:
        # Initialize the DataFrameProcessor with the product data path and key fields.
        processor = DataFrameProcessor(
            data_path="product_tables",
            key_field=key_field,
            location_field=location_field,
            fs=fs,
        )

        # Preprocess the data.
        processor.preprocess()

        # Save the processed tabular data as a pickle file.
        remove_object(fs=fs, object_name=tabular_object_name)
        save_object(fs=fs, object=processor.df, object_name=tabular_object_name)

        # Save the processed textual data as a pickle file.
        remove_object(fs=fs, object_name=textual_object_name)
        save_object(fs=fs, object=processor.df_text, object_name=textual_object_name)

        # Generate a profiling report for the tabular data.
        tabular_profile = ProfileReport(processor.df, title="Product Tabular Report")
        # Save the tabular profiling report to an HTML file.
        tabular_profile.to_file("reports/product-tabular-report.html")

        # Generate a profiling report for the textual data.
        textual_profile = ProfileReport(
            processor.df_text, title="Product Textual Report"
        )
        # Save the textual profiling report to an HTML file.
        textual_profile.to_file("reports/product-textual-report.html")

    else:
        print("Skipping processing and profiling.")


if __name__ == "__main__":
    main()
