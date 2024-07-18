#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import os
import pickle

import pandas as pd
import yaml
from tqdm import tqdm

from mongodb_lib import *
from openai_handlers import query_gpt_with_history

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def main():
    """
    Main function to classify product descriptions using GPT-4 OpenAI model and save results to MongoDB.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model.")
    parser.add_argument("--apikey", type=str, required=True, help="OpenAI API key.")

    args = parser.parse_args()

    model_name = args.model_name
    apikey = args.apikey

    object_name = f"product_textual_english_summarized_categories"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:

        # Import taxonomy
        taxonomy = pd.read_excel("Categories.xlsx")
        df_categories = taxonomy[
            ["Category", "Description & Keywords"]
        ].drop_duplicates()
        categories = [el.split(": ")[1] for el in df_categories["Category"]]
        categories_descriptions = [
            el.split("\n")[0].replace("- Description: ", "")
            for el in df_categories["Description & Keywords"]
        ]
        d = dict(zip(categories, categories_descriptions))

        # Import summarized texts
        data = read_object(fs, "product_textual_english_summarized")
        data = pd.DataFrame(data)
        data.fillna("", inplace=True)

        input_texts = [
            x + ": " + y
            for x, y in zip(
                data["PRODUCTCODE"].tolist(),
                data["pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED"].tolist(),
            )
        ]

        conversation_history = [
            {"role": "system", "content": "Hello! How can I assist you today?"}
        ]

        initial_prompt = (
            "You are a multi-label classifier specialized in analyzing product descriptions from online booking platforms, such as tours and activities. Your task is to classify each product description into a maximum of three relevant labels, listed in descending order of relevance. "
            "In the next prompts, I will send you a list of product descriptions, and for each product description, provide a Python-formatted list of the corresponding labels. If a product description has no text or have no corresponding labels, return an empty list for that element. "
            "Your response should only include a Python dictionary where the keys are the product codes, and their values are the labels. Do not include any additional text to your output."
            "Example input: ['100213P12: Saint-Malo - Bayeux Transfer, our professional english speaking drivers guarantee a punctual service available 7 days a week. Let you drive and travel in a luxurious and comfortable minivan Mercedes.', '100213P14: Mont Saint-Michel is located about 4 hours from Paris. The price is all include for a transfer up to 7 people. The service is available 7 days a week.']. "
            "Output: {'100213P12': ['Label 1', 'Label 2'], '100213P14': ['Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5']}. "
            f"Here is a dictionary of the labels and their descriptions: {d}. "
            "Are you ready to start?"
        )

        result = query_gpt_with_history(
            apikey, initial_prompt, model_name, conversation_history
        )
        result = result.choices[0].message.content
        conversation_history.append({"role": "user", "content": initial_prompt})
        conversation_history.append({"role": "system", "content": result})

        batch_size = 100
        batches = [
            input_texts[i : i + batch_size]
            for i in range(0, len(input_texts), batch_size)
        ]

        tmp_dir = "tmp"

        for batch_idx, batch in enumerate(tqdm(batches)):

            result = query_gpt_with_history(
                apikey, str(batch), model_name, conversation_history
            )

            r = result.choices[0].message.content

            with open(f"{tmp_dir}/batch_{batch_idx}_categories_gpt4o.pkl", "wb") as f:
                pickle.dump(r, f)

        final_d = {}

        for file_name in os.listdir(tmp_dir):

            file_path = os.path.join(tmp_dir, file_name)

            with open(file_path, "rb") as f:
                annotations = pickle.load(f)
                s = annotations.find("{")
                e = annotations.rfind("}")
                annotations = annotations[s : e + 1]
                annotations = annotations.replace("Valentine's Day", "Valentines Day")
                annotations_dict = ast.literal_eval(annotations)
                final_d.update(annotations_dict)

        l = []

        for pc in tqdm(data["PRODUCTCODE"].tolist()):
            if pc in list(final_d.keys()):
                l.append(final_d[pc])
            else:
                l.append([])

        data["categories_gpt4o"] = l

        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=data, object_name=object_name)


if __name__ == "__main__":
    main()
