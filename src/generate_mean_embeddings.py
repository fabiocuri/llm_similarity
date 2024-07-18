#!/usr/bin/env python
# coding: utf-8

import argparse
import gc
import pickle

import numpy as np
from sklearn.decomposition import PCA

from generate_model_embeddings import read_embedding

# Run garbage collection to free up memory.
gc.collect()


def reduce_dimension(concatenated_array, target_dim=1000):
    """
    Reduce the dimension of the concatenated array using PCA.

    Parameters:
    concatenated_array (np.ndarray): Array containing concatenated embeddings.
    target_dim (int): Target dimension after PCA reduction.

    Returns:
    np.ndarray: Array with reduced dimensions.
    """
    pca = PCA(n_components=target_dim)
    reduced_array = pca.fit_transform(concatenated_array)
    return reduced_array


def main():
    """
    Main function to reduce dimensions of concatenated model embeddings.

    Steps:
    1. Load embeddings from specified embedding models and fields.
    2. Concatenate the loaded embeddings.
    3. Reduce the dimensions of the concatenated embeddings using PCA.
    4. Save the reduced embeddings as a pickle file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Enable overwrite mode"
    )
    parser.add_argument(
        "--embedding_models",
        type=str,
        required=True,
        help="Comma-separated list of embedding models.",
    )
    parser.add_argument(
        "--embedding_fields", type=str, required=True, help="Embedding fields."
    )

    args = parser.parse_args()

    embedding_models = args.embedding_models.split(",")
    embedding_fields = args.embedding_fields

    object_name = f"model_embeddings_mean_{embedding_fields}"

    if args.overwrite:

        embeddings_list = []

        for em in embedding_models:
            model_name = em.split("/")[-1]
            em_path = f"tmp/model_embeddings_{model_name}_{embedding_fields}"
            embeddings_list.append(read_embedding(em_path))

        # Concatenate the arrays along the feature axis (axis=1)
        concatenated_array = np.concatenate(embeddings_list, axis=1)

        # Reduce dimensions to the target dimension
        reduced_array = reduce_dimension(concatenated_array)

        with open(f"tmp/{object_name}", "wb") as f:
            pickle.dump(reduced_array, f)

    else:
        print("Skipping generation of mean embeddings.")


if __name__ == "__main__":
    main()
