import argparse
import ast
import gc
import re
import time
from collections import defaultdict

import gspread
import pandas as pd
import yaml
from oauth2client.service_account import ServiceAccountCredentials
from tqdm import tqdm

from mongodb_lib import *
from openai_handlers import query_gpt_with_history

# Load MongoDB configuration from YAML file
config_infra = yaml.load(open("infra-config-pipeline.yaml"), Loader=yaml.FullLoader)
db, fs, client = connect_to_mongodb(config_infra)

# Run garbage collection to free up memory.
gc.collect()

system_role = "You are an expert in online bookings and product matching in the tourism and entertainment industry. Your expertise includes comparing product descriptions to identify highly similar products."


def preprocess_chunk_openai(df_product, chunk, fields_openai, text_field, title_field):

    df_product = df_product.astype(str)
    del df_product["TotalReviews"]
    chunk = chunk.astype(str)
    del chunk["TotalReviews"]

    if fields_openai == "title":

        del df_product[text_field]
        del chunk[text_field]

    if fields_openai == "text":

        del df_product[title_field]
        del chunk[title_field]

    product_features = "\n".join(
        [f"{col}: {list(df_product[col])[0]}" for col in list(df_product.columns)]
    )

    candidates_str = ""

    for _, row in chunk.iterrows():

        df_now = pd.DataFrame(row).T

        candidates_str_now = "\n".join(
            [f"{col}: {list(df_now[col])[0]}" for col in list(df_now.columns)]
        )

        candidates_str += "\n \n" + candidates_str_now

    return product_features, candidates_str


def append_to_google_sheets(credentials_file, results_out, file_name, sheet_name):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
    client = gspread.authorize(creds)

    # Open the Google Sheet
    sheet = client.open(file_name).worksheet(sheet_name)

    # Append data

    for line in results_out:

        if line:

            if isinstance(line[0], list):

                for l_ in line:

                    sheet.append_row(l_)

            else:

                sheet.append_row(line)

        time.sleep(10)


def range_to_tuple(range_str):
    if not range_str:
        return (float("-inf"), float("-inf"))
    parts = range_str.strip("()[]").split(",")
    return (float(parts[0]), float(parts[1]))


def landmarks_are_the_same(list1, list2):

    return list(sorted(list1)) == list(sorted(list2))


def main():

    parser = argparse.ArgumentParser(
        description="Process product similarity experiment parameters."
    )

    # Add arguments
    parser.add_argument(
        "-credentials",
        required=True,
        help="Path to Google Sheets credentials JSON file",
    )
    parser.add_argument(
        "-product_id", type=str, required=True, help="The ID of the product."
    )
    parser.add_argument(
        "-city_name", type=str, required=True, help="The name of the city."
    )
    parser.add_argument(
        "-supplier_code", type=str, required=True, help="Supplier code."
    )
    parser.add_argument(
        "-average_rating", type=str, required=True, help="Tour average rating."
    )
    parser.add_argument(
        "-start_year", type=str, required=True, help="Star year of products."
    )
    parser.add_argument(
        "-landmarks", type=str, required=True, help="Landmarks of the product."
    )
    parser.add_argument(
        "-is_private",
        type=str,
        required=True,
        help="Whether the activity is private or not.",
    )
    parser.add_argument(
        "-categories",
        type=str,
        required=True,
        help="Categories of Walkway AI's taxonomy.",
    )
    parser.add_argument(
        "-embedding_fields", type=str, required=True, help="Embedding fields."
    )
    parser.add_argument("-apikey", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("-experiment_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("-chunk_size", type=str, required=True, help="chunk_size")
    parser.add_argument("-openai_model", type=str, required=True, help="openai_model")
    parser.add_argument("-fields_openai", type=str, required=True, help="fields_openai")

    args = parser.parse_args()

    chunk_size = int(args.chunk_size)
    openai_model = args.openai_model
    fields_openai = args.fields_openai

    if args.embedding_fields == "Product Description, Product Title":

        args.embedding_fields = "description_title"

    elif (
        args.embedding_fields == "Product Title, Incl/Excl Text, Tour Grade Description"
    ):

        args.embedding_fields = "title_inclexcl_tgdescription"

    elif args.embedding_fields == "Product Description, Incl/Excl Text":

        args.embedding_fields = "description_inclexcl"

    elif (
        args.embedding_fields
        == "Product Description, Product Title, Incl/Excl Text, Tour Grade Description"
    ):

        args.embedding_fields = "title_inclexcl_tgdescription_description"

    else:

        ValueError("Option not valid.")

    city_feature = "pdt_product_detail_VIDESTINATIONCITY"
    supplier_code_feature = "pdt_product_level_SUPPLIERCODE"
    avg_rating_feature = "pdt_product_level_TOTALAVGRATING"
    time_feature = "bookings_MOSTRECENTORDERDATE"
    private_feature = "pdt_product_level_ISPRIVATETOUR"
    text_field = "pdt_product_detail_PRODUCTDESCRIPTION_SUMMARIZED"
    product_field = "PRODUCTCODE"
    title_field = "pdt_product_detail_PRODUCTTITLE_translated"

    test_products = yaml.load(open("test_products.yaml"), Loader=yaml.FullLoader)

    object_name = f"product_similarities_mean_{args.embedding_fields}"
    existing_file = fs.find_one({"filename": object_name})

    run_openai = True

    if existing_file:

        product_ids = [el.strip() for el in args.product_id.split(",")]

        columns_results = [
            [
                "Chunk size",
                "Open AI Model",
                "Fields Open AI",
                "% wo OpenAI",
                "% w OpenAI",
                "N. wo OpenAI",
                "N. w OpenAI",
                "OpenAI Score",
                "N. Mandatory Matches",
                "Missing after OpenAI",
            ]
        ]

        append_to_google_sheets(
            args.credentials,
            columns_results,
            "Scores - Product Similarity 24",
            "OpenAI",
        )

        for product_id in product_ids:

            args.product_id = product_id

            similarity_dict = read_object(fs, object_name)
            similar_products = similarity_dict[args.product_id]
            id_score = defaultdict(lambda: 0)
            id_score.update({key: value for key, value in similar_products})

            all_products = list(id_score.keys())
            all_products.append(args.product_id)

            df_raw = read_object(fs, "product_tabular")
            df_raw = pd.DataFrame(df_raw)

            df_raw_possible = df_raw[df_raw[product_field].isin(all_products)]

            df_text = read_object(fs, "product_textual_english_summarized")
            df_text = pd.DataFrame(df_text)
            df_text_possible = df_text[df_text[product_field].isin(all_products)]

            df = pd.merge(
                df_raw_possible, df_text_possible, on=product_field, how="outer"
            )

            df = df[
                [
                    product_field,
                    text_field,
                    city_feature,
                    supplier_code_feature,
                    time_feature,
                    private_feature,
                    title_field,
                ]
            ]

            # Product features
            df_product = df[df[product_field] == args.product_id]

            # Candidate features
            df = df[df[product_field] != args.product_id]

            # Sort by scores
            df["score"] = [id_score[p_id] for p_id in list(df[product_field])]
            df["score"] = df["score"].astype(float)
            df = df.sort_values(by="score", ascending=False)
            del df["score"]

            print(f"Selected product: {args.product_id}")

            # print(f"Number of initial candidates: {df.shape[0]}")

            ## CITY FILTER

            if args.city_name == "same":

                df = df[df[city_feature] == str(list(df_product[city_feature])[0])]

            # print(f"Number of candidates after the city filter: {df.shape[0]}")

            ## SUPPLIER CODE FILTER

            if args.supplier_code == "different":

                df = df[
                    df[supplier_code_feature]
                    != str(list(df_product[supplier_code_feature])[0])
                ]

            # print(f"Number of candidates after the supplier code filter: {df.shape[0]}")

            ## AVERAGE RATING FILTER

            if args.average_rating == "similar":

                reviews_table = read_object(fs, "reviews_product")
                reviews_table = pd.DataFrame(reviews_table)

                mapping_2_avgrating = defaultdict(
                    lambda: 0,
                    zip(
                        reviews_table["ProductCode"],
                        reviews_table["AVGRating"],
                    ),
                )

                product_avg_rating = mapping_2_avgrating[args.product_id]

                df[avg_rating_feature] = [
                    mapping_2_avgrating[el] for el in list(df[product_field])
                ]

                tolerance = 0.1 * product_avg_rating
                avg_bool = [
                    abs(product_avg_rating - float(x)) <= tolerance
                    for x in list(df[avg_rating_feature])
                ]
                df = df[avg_bool]
                del df[avg_rating_feature]

            # print(f"Number of candidates after the average rating filter: {df.shape[0]}")

            ## START YEAR FILTER

            if args.start_year != "any":

                df["year"] = pd.to_datetime(df[time_feature], unit="ms")
                df["year"] = df["year"].dt.year

                df = df[df["year"] >= int(args.start_year)]
                del df["year"]

            # print(f"Number of candidates after the year filter: {df.shape[0]}")

            ## LANDMARKS FILTER

            d_landmarks = {}

            one_hot_encoding = read_object(fs, "one_hot_encoding_landmarks")
            name_landmarks = read_object(fs, "name_landmarks")
            list_products = list(df_raw[product_field])

            idx_product = list_products.index(args.product_id)
            which_landmarks = one_hot_encoding[idx_product]
            which_landmarks = [bool(x) for x in which_landmarks]
            names_landmarks_product = [
                elem for elem, flag in zip(name_landmarks, which_landmarks) if flag
            ]

            if names_landmarks_product:

                d_landmarks[args.product_id] = names_landmarks_product

                if args.landmarks == "same":

                    final_candidates = list()

                    for candidate in list(df[product_field]):

                        idx_candidate = list_products.index(candidate)
                        which_landmarks = one_hot_encoding[idx_candidate]
                        which_landmarks = [bool(x) for x in which_landmarks]
                        names_landmarks_candidate = [
                            elem
                            for elem, flag in zip(name_landmarks, which_landmarks)
                            if flag
                        ]
                        result = landmarks_are_the_same(
                            names_landmarks_product, names_landmarks_candidate
                        )

                        if result:

                            final_candidates.append(candidate)
                            d_landmarks[candidate] = names_landmarks_candidate

                    df = df[df[product_field].isin(final_candidates)]

            # print(f"Number of candidates after the landmarks filter: {df.shape[0]}")

            ## PRIVATE OPTION FILTER

            if args.is_private == "same":

                df = df[
                    df[private_feature] == str(list(df_product[private_feature])[0])
                ]

            # print(f"Number of candidates after the private filter: {df.shape[0]}")

            ## CATEGORY FILTER

            annotated_data = read_object(
                fs, "product_textual_english_summarized_categories_walkway"
            )
            annotated_data = pd.DataFrame(annotated_data)
            annotated_data = annotated_data[[product_field, "categories-walkway"]]
            annotated_data = annotated_data.set_index(product_field)[
                "categories-walkway"
            ].to_dict()

            if args.categories != "any":

                product_categories = annotated_data[args.product_id]

                if product_categories:

                    l_pd = list()

                    for prd_ in list(df[product_field]):

                        if args.categories == "same":

                            if set(product_categories).issubset(
                                set(annotated_data[prd_])
                            ):

                                l_pd.append(prd_)

                        if args.categories == "one":

                            if any(
                                ct in annotated_data[prd_] for ct in product_categories
                            ):

                                l_pd.append(prd_)

                    df = df[df[product_field].isin(l_pd)]

            # print(f"Number of candidates after the category filter: {df.shape[0]}")

            ## REVIEWS FILTER (sorted)

            reviews_table = read_object(fs, "reviews_product")
            reviews_table = pd.DataFrame(reviews_table)

            mapping_2_totalreviews = defaultdict(
                lambda: 0,
                zip(
                    reviews_table["ProductCode"],
                    reviews_table["TotalReviews"],
                ),
            )

            df["TotalReviews"] = [
                mapping_2_totalreviews[el] for el in df[product_field]
            ]
            df = df.sort_values(by="TotalReviews", ascending=False)

            df_product["TotalReviews"] = [mapping_2_totalreviews[args.product_id]]

            print(f"Final number of candidates: {df.shape[0]}")

            del df[city_feature]
            del df[supplier_code_feature]
            del df[time_feature]
            del df[private_feature]

            del df_product[city_feature]
            del df_product[supplier_code_feature]
            del df_product[time_feature]
            del df_product[private_feature]

            # Create selected product summary

            output_product_categories = list(set(annotated_data[args.product_id]))

            product_features = "\n".join(
                [
                    f"{col}: {list(df_product[col])[0]}"
                    for col in list(df_product.columns)
                ]
            )
            product_features = product_features.replace(
                text_field, "Summarized description"
            )
            product_features = product_features.replace(title_field, "Title")

            product_features = product_features.replace(title_field, "Title")
            product_features = (
                product_features + "\nCategory: " + str(output_product_categories)
            )

            # Raw results

            df_no_openai = df

            result_features_wo_openai = list()

            for _, row in df_no_openai.iterrows():

                df_now = pd.DataFrame(row).T

                product_id = list(df_now[product_field])[0]
                no_openai_product_categories = list(set(annotated_data[product_id]))

                result_features = "\n".join(
                    [f"{col}: {list(df_now[col])[0]}" for col in list(df_now.columns)]
                )
                result_features = result_features.replace(
                    text_field,
                    "Summarized description",
                )
                result_features = result_features.replace(
                    title_field,
                    "Title",
                )
                result_features = (
                    result_features + "\nCategory: " + str(no_openai_product_categories)
                )

                result_features_wo_openai.append(result_features.split("\n"))

            # Filter results with OpenAI

            result_features_w_openai = list()

            if run_openai:

                conversation_history = [
                    {"role": "system", "content": "Hello! How can I assist you today?"}
                ]

                initial_prompt = """
                You are an expert in finding similar products from the tourism and activities industry.
                In the next prompts, I will send you a REFERENCE PRODUCT and a list of CANDIDATE PRODUCTS. To be considered almost identical to the REFERENCE PRODUCT, a CANDIDATE PRODUCT must:

                1. Visit the same destination, landmark, or region.
                2. Offer a similar type of experience, including:
                - Same tour route or access level.
                - Similar activities (e.g., food, private transportation, etc.).
                - Same category of experience (e.g., private tours, walking tours, museum tours).
                - Customization options (e.g., personalized itineraries, private groups).
                - Type of transportation (e.g., shuttle, bike, walking).
                - Duration of the experience (e.g., 2 hours, half-day, full-day).
                - Key landmarks or attractions visited.
                - Type of guide (e.g., local guide, professional driver).
                - Additional features (e.g., skip-the-line access, hotel pick-up).

                Not all of the conditions above need to be respected, but keep them in mind when deciding.
                In your answer, return ONLY a Python list of the CANDIDATE PRODUCTS that you consider similar to the REFERENCE PRODUCT (e.g., ['18745FBP', 'H73TOUR2']). If there are no similar products, return an empty list ([]).
                Are you ready to start?
                """

                # NEXT: add logic where we have several levels of similarity (high/medium/low)

                result = query_gpt_with_history(
                    args.apikey, initial_prompt, openai_model, conversation_history
                )
                result = result.choices[0].message.content
                conversation_history.append({"role": "user", "content": initial_prompt})
                conversation_history.append({"role": "system", "content": result})

                df_openai = df

                chunks = [
                    df_openai.iloc[i : i + chunk_size]
                    for i in range(0, len(df_openai), chunk_size)
                ]

                result_all_chunks = list()

                for chunk in tqdm(chunks, total=len(chunks)):

                    finished_check = True

                    product_features, candidates_str = preprocess_chunk_openai(
                        df_product, chunk, fields_openai, text_field, title_field
                    )
                    product_features = product_features.replace(
                        "PRODUCTCODE: ", "REFERENCE PRODUCT: "
                    )
                    candidates_str = candidates_str.replace(
                        "PRODUCTCODE: ", "CANDIDATE PRODUCT: "
                    )
                    prompt_now = product_features + "\n" + candidates_str

                    while finished_check:

                        result = query_gpt_with_history(
                            args.apikey, prompt_now, openai_model, conversation_history
                        )

                        try:

                            result = re.findall(
                                r"\[.*?\]", result.choices[0].message.content
                            )[0]
                            result = ast.literal_eval(result)
                            result_all_chunks.append(result)
                            finished_check = False

                        except Exception as e:

                            print(result)
                            print(e)

                result_all_chunks = [
                    item for sublist in result_all_chunks for item in sublist
                ]

                if len(result_all_chunks) > 0:

                    df_openai = df_openai[
                        df_openai[product_field].isin(result_all_chunks)
                    ]

                    for _, row in df_openai.iterrows():

                        df_now = pd.DataFrame(row).T

                        product_id = list(df_now[product_field])[0]
                        openai_product_categories = list(
                            set(annotated_data[product_id])
                        )

                        result_features = "\n".join(
                            [
                                f"{col}: {list(df_now[col])[0]}"
                                for col in list(df_now.columns)
                            ]
                        )
                        result_features = result_features.replace(
                            text_field,
                            "Summarized description",
                        )
                        result_features = result_features.replace(
                            title_field,
                            "Title",
                        )

                        result_features = (
                            result_features
                            + "\nCategory: "
                            + str(openai_product_categories)
                        )

                        result_features_w_openai.append(result_features.split("\n"))

            columns_results = [
                "Experiment ID",
                "City",
                "Supplier Code",
                "Average Rating",
                "Start Year",
                "Landmarks",
                "Private",
                "Categories",
                "Embedding fields",
            ]

            exp_params = [
                args.experiment_id,
                args.city_name,
                args.supplier_code,
                args.average_rating,
                args.start_year,
                args.landmarks,
                args.is_private,
                args.categories,
                args.embedding_fields,
            ]

            results_out = [
                columns_results,
                exp_params,
                product_features.split("\n"),
                ["SIMILAR PRODUCTS WITHOUT OPENAI"],
                result_features_wo_openai,
                ["SIMILAR PRODUCTS WITH OPENAI"],
                result_features_w_openai,
                ["*****"],
            ]

            append_to_google_sheets(
                args.credentials,
                results_out,
                "WalkwayAI - Product Similarity",
                "Sheet1",
            )

            file_path = f"experiment_results/{args.product_id}.xlsx"

            df_result_out = pd.DataFrame(columns=columns_results)

            df_result_out.loc[0] = exp_params

            def pad_row(row, length):
                return row + [None] * (length - len(row))

            row_index = 1

            for feature in product_features.split("\n"):
                df_result_out.loc[row_index] = pad_row([feature], len(columns_results))
                row_index += 1

            df_result_out.loc[row_index] = pad_row(["\n"], len(columns_results))
            row_index += 1

            df_result_out.loc[row_index] = pad_row(
                ["SIMILAR PRODUCTS WITHOUT OPENAI"], len(columns_results)
            )
            row_index += 1

            for feature in result_features_wo_openai:
                for ff in feature:
                    df_result_out.loc[row_index] = pad_row([ff], len(columns_results))
                    row_index += 1

            df_result_out.loc[row_index] = pad_row(["\n"], len(columns_results))
            row_index += 1

            df_result_out.loc[row_index] = pad_row(
                ["SIMILAR PRODUCTS WITH OPENAI"], len(columns_results)
            )
            row_index += 1

            for feature in result_features_w_openai:
                for ff in feature:
                    df_result_out.loc[row_index] = pad_row([ff], len(columns_results))
                    row_index += 1

            df_result_out.loc[row_index] = pad_row(["*****"], len(columns_results))

            df_result_out.to_excel(file_path, index=False)

            gc.collect()

            # Calculate score for this product
            mandatory_similar_products_original = test_products[args.product_id]
            mandatory_similar_products_original = (
                mandatory_similar_products_original.split(",")
            )
            mandatory_similar_products_original = [
                el.strip() for el in mandatory_similar_products_original
            ]
            possible_similar_products = list(
                set(
                    [
                        el
                        for el in mandatory_similar_products_original
                        if el in list(df_raw[product_field])
                    ]
                )
            )
            n_possible_mandatory_products = len(possible_similar_products)
            n = n_possible_mandatory_products

            str_wo_openai = [
                el[0].split(":")[1].strip() for el in result_features_wo_openai
            ]
            str_w_openai = [
                el[0].split(":")[1].strip() for el in result_features_w_openai
            ]

            c_wo_openai, c_w_openai = 0, 0

            l_missing_openai = list()

            for msp in possible_similar_products:

                if msp in str_wo_openai:

                    c_wo_openai += 1

                if msp in str_w_openai:

                    c_w_openai += 1

                else:

                    l_missing_openai.append(msp)

            pctg_wo_openai = c_wo_openai * 100 / n
            pctg_w_openai = c_w_openai * 100 / n
            n_candidates_raw = len(df_no_openai)
            n_candidates_openai = len(df_openai)
            openai_score = pctg_w_openai / n_candidates_openai

            results_scores = [
                [
                    chunk_size,
                    openai_model,
                    fields_openai,
                    pctg_wo_openai,
                    pctg_w_openai,
                    n_candidates_raw,
                    n_candidates_openai,
                    openai_score,
                    n,
                    str(l_missing_openai),
                ]
            ]

            append_to_google_sheets(
                args.credentials,
                results_scores,
                "Scores - Product Similarity 24",
                "OpenAI",
            )


if __name__ == "__main__":
    main()
