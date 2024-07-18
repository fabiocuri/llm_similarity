import argparse
import gc
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from mongodb_lib import *

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def find_mentioned_landmarks(description, candidates):
    """
    Finds landmarks mentioned in the description based on given candidates.

    Args:
        description (str): The text description to search.
        candidates (dict): Dictionary of variations of landmarks to search for.

    Returns:
        list: List of mentioned landmarks found in the description.
    """
    description = remove_accents(description).lower()

    mentioned_landmarks = []

    for landmark in candidates:
        # Check if the landmark or its aliases are mentioned in the description
        if landmark in description:
            mentioned_landmarks.append(landmark)
        else:
            # Add fuzzy matching or other checks as needed for variations or typos
            pass

    return mentioned_landmarks


def remove_accents(input_str):
    """
    Removes accents from characters in the input string.

    Args:
        input_str (str): The input string containing accented characters.

    Returns:
        str: String without accents.
    """
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def generate_variations(keyword):
    """
    Generates variations of a keyword including its alternative names in parentheses.

    Args:
        keyword (str): The original keyword possibly containing alternative names.

    Returns:
        set: Set of variations (lowercase) of the keyword.
    """
    variations = set()

    # Original keyword
    variations.add(keyword.lower())
    variations.add(remove_accents(keyword).lower())

    # Alternative names in parentheses
    match = re.match(r"(.+)\((.+)\)", keyword)

    if match:
        base_name = match.group(1).strip()
        alt_name = match.group(2).strip()
        variations.add(base_name.lower())
        variations.add(remove_accents(base_name).lower())
        variations.add(alt_name.lower())
        variations.add(remove_accents(alt_name).lower())

    return variations


def flatten_dict(d):
    """
    Flattens a nested dictionary into a list of sorted unique values.

    Args:
        d (dict): The nested dictionary to flatten.

    Returns:
        list: List of sorted unique values from the dictionary.
    """
    flattened_keys = []

    for city in d.keys():
        for name in list(d[city].values()):
            flattened_keys.append(name)

    return list(sorted(set(flattened_keys)))


def main():
    """
    Main function to perform landmark detection and save results to MongoDB.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    args = parser.parse_args()

    object_name_one_hot_encoding = "one_hot_encoding_landmarks"
    existing_file_one_hot_encoding = fs.find_one(
        {"filename": object_name_one_hot_encoding}
    )

    object_name_landmarks = "name_landmarks"
    existing_file_name_landmarks = fs.find_one({"filename": object_name_landmarks})

    if (
        not existing_file_one_hot_encoding
        or not existing_file_name_landmarks
        or args.overwrite
    ):

        df = read_object(fs, "product_tabular")
        df_text_sum = read_object(fs, "product_textual_english")

        df = pd.DataFrame(df)
        df_text_sum = pd.DataFrame(df_text_sum)

        df.fillna("", inplace=True)
        df_text_sum.fillna("", inplace=True)

        assert list(df["PRODUCTCODE"]) == list(df_text_sum["PRODUCTCODE"])

        landmarks = yaml.load(open("landmarks.yaml"), Loader=yaml.FullLoader)

        variations_dict = defaultdict(lambda: defaultdict(list))

        for city in landmarks["destinations"]:
            candidates = landmarks["destinations"].get(city, {}).get("landmarks", [])

            # Generate variations for each place and populate the dictionary
            for place in candidates:
                variations = generate_variations(place)
                for variation in variations:
                    variations_dict[city][variation] = place

        all_landmarks = flatten_dict(variations_dict)
        one_hot_encoding = []

        for city, text_summarized in tqdm(
            zip(
                list(df["pdt_product_detail_VIDESTINATIONCITY"]),
                list(df_text_sum["pdt_product_detail_PRODUCTDESCRIPTION_translated"]),
            ),
            total=len(df),
        ):

            candidates = variations_dict[city]

            mentioned_landmarks = find_mentioned_landmarks(text_summarized, candidates)
            mentioned_landmarks = list(
                sorted(set([candidates[el] for el in mentioned_landmarks]))
            )

            one_hot_encoding_now = []

            for landmark in all_landmarks:
                if landmark in mentioned_landmarks:
                    one_hot_encoding_now.append(1)
                else:
                    one_hot_encoding_now.append(0)

            one_hot_encoding.append(one_hot_encoding_now)

        one_hot_encoding = np.array(one_hot_encoding)

        remove_object(fs=fs, object_name=object_name_one_hot_encoding)
        save_object(
            fs=fs, object=one_hot_encoding, object_name=object_name_one_hot_encoding
        )

        remove_object(fs=fs, object_name=object_name_landmarks)
        save_object(fs=fs, object=all_landmarks, object_name=object_name_landmarks)


if __name__ == "__main__":
    main()
