import argparse
import gc

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from generate_model_embeddings import read_embedding
from mongodb_lib import *

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()


def find_most_similar_products(embedding, embeddings, num_similar=50):
    """
    Find the most similar products based on cosine similarity.

    Parameters:
    embedding (np.ndarray): The embedding of the given product.
    embeddings (np.ndarray): The array of embeddings for all products.
    num_similar (int): The number of similar products to find.

    Returns:
    tuple: Indices of the most similar products and their similarity scores.
    """
    embedding = embedding.reshape(1, -1)
    similarities = cosine_similarity(embedding, embeddings)[0]
    similar_indices = similarities.argsort()[-(num_similar + 1) : -1][::-1]
    similar_scores = similarities[similar_indices]
    return similar_indices, similar_scores


def main():
    """
    Main function to find the most similar products based on embeddings.

    Steps:
    1. Check if the similarity data already exists in MongoDB.
    2. If not, load the product textual data and combined embeddings.
    3. Calculate the most similar products for each product in the dataset.
    4. Save the similarity data to MongoDB.
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
    object_name = f"product_similarities_{model_name}_{embedding_fields}"
    existing_file = fs.find_one({"filename": object_name})

    if not existing_file or args.overwrite:
        df = read_object(fs, "product_textual_english_summarized")
        df = pd.DataFrame(df)
        combined_embeddings = read_embedding(
            f"tmp/model_embeddings_{model_name}_{embedding_fields}"
        )
        combined_embeddings = np.array(combined_embeddings)
        similarity_dict = {}

        for given_product_index in tqdm(
            range(len(df["PRODUCTCODE"])), desc="Calculating similarities"
        ):
            given_embedding = combined_embeddings[given_product_index]
            most_similar_indices, similarity_scores = find_most_similar_products(
                given_embedding, combined_embeddings, num_similar=200
            )
            similar_products = [
                df["PRODUCTCODE"].iloc[el] for el in most_similar_indices
            ]
            similarity_scores = [str(x) for x in similarity_scores]
            similarity_dict[df["PRODUCTCODE"].iloc[given_product_index]] = list(
                zip(similar_products, similarity_scores)
            )

        remove_object(fs=fs, object_name=object_name)
        save_object(fs=fs, object=similarity_dict, object_name=object_name)

    else:
        print("Skipping product similarity.")


if __name__ == "__main__":
    main()
