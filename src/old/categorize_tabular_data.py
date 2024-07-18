#!/usr/bin/env python
# coding: utf-8

import argparse
import gc

import pandas as pd
import yaml

from mongodb_lib import *

# Load configuration from YAML files.
config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
bigquery_config = config["bigquery-to-retrieve"]
key_field = bigquery_config["key-field"]

config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def main():
    """
    Main function to categorize product data.

    Steps:
    1. Load the tabular product data from a pickle file.
    2. Create a copy of the DataFrame for categorization.
    3. Factorize each column in the DataFrame.
    4. Remove the key field and supplier code from the categorized DataFrame.
    5. Save the categorized DataFrame as a pickle file if it does not already exist in MongoDB.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )

    args = parser.parse_args()

    object_name = "product_tabular_categorized"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:

        # Load the tabular product data from a pickle file.
        df = read_object(fs, "product_tabular")
        df = pd.DataFrame(df)

        # Factorize each column in the DataFrame.
        for col in df.columns:
            df[col], _ = pd.factorize(df[col])

        # Remove the key field from the categorized DataFrame.
        del df[key_field]

        # Remove the supplier code from the categorized DataFrame.
        del df["pdt_product_level_SUPPLIERCODE"]

        # Remove the date from the categorized DataFrame.
        del df["bookings_MOSTRECENTORDERDATE"]

        # Save the categorized DataFrame as a pickle file.
        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=df, object_name=object_name)
    else:
        print("Skipping categorization of tabular data.")


if __name__ == "__main__":
    main()
