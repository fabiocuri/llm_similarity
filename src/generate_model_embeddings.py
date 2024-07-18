#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import pickle

import numpy as np

# Run garbage collection to free up memory.
gc.collect()


def read_embedding(path):
    """
    Read embeddings from a pickle file.

    Parameters:
    path (str): Path to the pickle file containing embeddings.

    Returns:
    object: Loaded embeddings object.
    """
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def main():
    """
    Main function to concatenate tabular data with text embeddings.

    Steps:
    1. Load the categorized tabular data from a pickle file.
    2. Load the embeddings for 'pdt_inclexcl_ENG_CONTENT'.
    3. Load the embeddings for 'pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED'.
    4. Concatenate the tabular data with the two sets of embeddings.
    5. Save the final concatenated embeddings as a pickle file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    parser.add_argument(
        "--embedding_model", type=str, required=True, help="The embedding model."
    )
    parser.add_argument(
        "--embedding_fields", type=str, required=True, help="Embedding fields."
    )

    args = parser.parse_args()

    embedding_model = args.embedding_model
    model_name = embedding_model.split("/")[-1]

    embedding_fields = args.embedding_fields

    object_name = f"model_embeddings_{model_name}_{embedding_fields}"

    if args.overwrite:

        embeddings = []

        if embedding_fields in [
            "description_title",
            "description_inclexcl",
            "title_inclexcl_tgdescription_description",
        ]:
            embeddings.append(
                np.array(
                    read_embedding(
                        f"tmp/embeddings_pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED_{model_name}"
                    )
                )
            )

        if embedding_fields in [
            "title_inclexcl_tgdescription",
            "description_inclexcl",
            "title_inclexcl_tgdescription_description",
        ]:
            embeddings.append(
                np.array(
                    read_embedding(
                        f"tmp/embeddings_pdt_inclexcl_ENG_CONTENT_translated_{model_name}"
                    )
                )
            )

        if embedding_fields in [
            "title_inclexcl_tgdescription",
            "description_title",
            "title_inclexcl_tgdescription_description",
        ]:
            embeddings.append(
                np.array(
                    read_embedding(
                        f"tmp/embeddings_pdt_product_detail_PRODUCTTITLE_translated_{model_name}"
                    )
                )
            )

        if embedding_fields in [
            "title_inclexcl_tgdescription",
            "title_inclexcl_tgdescription_description",
        ]:
            embeddings.append(
                np.array(
                    read_embedding(
                        f"tmp/embeddings_pdt_product_detail_TOURGRADEDESCRIPTION_{model_name}"
                    )
                )
            )

        final_embeddings = np.concatenate(embeddings, axis=1)

        with open(f"tmp/{object_name}", "wb") as f:
            pickle.dump(final_embeddings, f)

    else:
        print("Skipping generation of model embeddings.")


if __name__ == "__main__":
    main()
