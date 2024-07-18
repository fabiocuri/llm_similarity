#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import os

import pandas as pd
import yaml
from tqdm import tqdm
from transformers import pipeline

from mongodb_lib import *

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def main():
    """
    Main function to summarize product descriptions.

    Steps:
    1. Load the product textual data from MongoDB.
    2. Check for intermediate results to potentially resume from.
    3. Initialize the summarization pipeline.
    4. Summarize each product description and save intermediate results.
    5. Save the final summarized descriptions to MongoDB.
    6. Clean up any intermediate files.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    parser.add_argument(
        "--summarization_model",
        type=str,
        required=True,
        help="The summarization model.",
    )

    args = parser.parse_args()

    summarization_model = args.summarization_model

    # Specify the name for the processed textual data in English.
    object_name = "product_textual_english_summarized"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:

        # Load the product textual data from MongoDB.
        df = read_object(fs, "product_textual_english")
        df = pd.DataFrame(df)
        df.fillna("", inplace=True)

        # Define the intermediate file path.
        intermediate_file = "tmp/product_textual_english_summarized_intermediate.pickle"

        # Check if an intermediate file exists to resume from.
        if os.path.exists(intermediate_file):
            df_intermediate = pd.read_pickle(intermediate_file)
            descriptions_summarized = df_intermediate[
                "pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED"
            ].tolist()
            start_index = len(descriptions_summarized)
            print(f"Resuming from index {start_index}")
        else:
            descriptions_summarized = []
            start_index = 0

        # Initialize the summarization pipeline.
        summarizer = pipeline("summarization", model=summarization_model)

        # Define the interval for saving intermediate results.
        save_interval = 50

        # Loop through each product description, starting from the last saved index.
        for i, desc in enumerate(
            tqdm(
                df["pdt_product_detail_PRODUCTDESCRIPTION_translated"][start_index:],
                total=len(df) - start_index,
                desc="Summarizing",
            )
        ):
            if desc:
                summarized_desc = summarizer(
                    desc, max_length=150, min_length=30, do_sample=False
                )[0]["summary_text"]
            else:
                summarized_desc = ""

            descriptions_summarized.append(summarized_desc)

            # Save intermediate results at defined intervals or at the end.
            if (i + 1) % save_interval == 0 or (start_index + i + 1) == len(df):
                df_intermediate = df.copy()
                df_intermediate = df_intermediate.iloc[: start_index + i + 1]
                df_intermediate[
                    "pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED"
                ] = descriptions_summarized
                df_intermediate.to_pickle(intermediate_file)
                print(f"Saved intermediate results at iteration {start_index + i + 1}")

        # Add the summarized descriptions to the DataFrame and save the final results.
        df["pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED"] = descriptions_summarized

        # Save the processed DataFrame to MongoDB.

        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=df, object_name=object_name)
        print("Saved final results")

    else:
        print("Skipping summarization.")


if __name__ == "__main__":
    main()
