#!/usr/bin/env python
# coding: utf-8

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm

from mongodb_lib import *

# Run garbage collection to free up memory.
gc.collect()


class DataFrameProcessor:
    def __init__(self, data_path, key_field, location_field, fs):
        """
        Initialize the DataFrameProcessor with a path to the pickle file.

        Parameters:
        data_path (str): The path to the pickle file containing the DataFrame.
        key_field (str): The column name to group by.
        location_field (str): The column name that contains the location.
        fs (GridFS): The GridFS object to interact with MongoDB.
        """
        self.data_path = data_path
        self.key_field = key_field
        self.location_field = location_field
        self.fs = fs

    def preprocess(self):
        """
        Preprocess the DataFrame by performing several steps including reading data,
        exploding columns, filling missing values, merging city data, aggregating fields,
        creating text DataFrame, and asserting sizes.
        """
        self.read_data()
        self.explode_dataframe()
        self.fill_missing_values()
        self.merge_and_remove_empty_cities()
        self.aggregate_all_fields()
        self.create_text_dataframe()
        self.preprocess_tabular_fields()
        self.preprocess_text_fields()
        self.assert_sizes()
        self.astypestr()

    def read_data(self):
        """
        Read the data from MongoDB GridFS and convert it to a DataFrame.
        """
        self.df = read_object(self.fs, self.data_path)
        self.df = pd.DataFrame(self.df)
        self.df_text = None

    def explode_dataframe(self):
        """
        Explode all columns of the DataFrame except the key field.
        """
        for col in tqdm(self.df.columns):
            if col != self.key_field:
                self.df = self.df.explode(col)

    def fill_missing_values(self):
        """
        Fill missing values in the DataFrame with NaN.
        """
        self.df.fillna(np.nan, inplace=True)

    def merge_and_remove_empty_cities(self):
        """
        Merge city values from multiple columns and drop rows where the location field is NaN.
        """

        def fill_city(row):
            if pd.isna(row[self.location_field]):
                if not pd.isna(row["pdt_product_level_VIDESTINATIONCITY"]):
                    return row["pdt_product_level_VIDESTINATIONCITY"]
                elif not pd.isna(row["pdt_inclexcl_ENG_VIDESTINATIONCITY"]):
                    return row["pdt_inclexcl_ENG_VIDESTINATIONCITY"]
                else:
                    return np.nan
            else:
                return row[self.location_field]

        self.df[self.location_field] = self.df.apply(fill_city, axis=1)

        del self.df["pdt_product_level_VIDESTINATIONCITY"]
        del self.df["pdt_inclexcl_ENG_VIDESTINATIONCITY"]

        self.df = self.df.dropna(subset=[self.location_field])

    def aggregate_all_fields(self):
        """
        Aggregate tabular fields in the DataFrame.
        """
        columns_to_aggregate = [col for col in self.df.columns if col != self.key_field]
        agg_dict = {col: list for col in columns_to_aggregate}
        self.df = self.df.groupby(self.key_field).agg(agg_dict).reset_index()

    def create_text_dataframe(self):
        """
        Create a separate DataFrame with descriptive content fields.
        """
        self.df_text = self.df.copy()
        self.descriptive_fields = [
            "pdt_inclexcl_ENG_CONTENT",
            "pdt_product_detail_PRODUCTDESCRIPTION",
            "pdt_product_detail_PRODUCTTITLE",
            "pdt_product_detail_TOURGRADEDESCRIPTION",
        ]
        self.df_text = self.df_text[[self.key_field] + self.descriptive_fields]

        for del_col in self.descriptive_fields:
            del self.df[del_col]

    def preprocess_tabular_fields(self):
        """
        Preprocess tabular fields in the DataFrame by reducing to unique values.
        """
        columns_to_aggregate = [col for col in self.df.columns if col != self.key_field]

        def get_unique_value(lst):
            unique_values = pd.unique(lst)
            unique_values = [x for x in unique_values if pd.notna(x)]
            if len(unique_values) == 1:
                return unique_values[0]
            elif len(unique_values) == 0:
                return np.nan
            else:
                return "; ".join(unique_values)

        for col in tqdm(columns_to_aggregate):
            self.df[col] = self.df[col].apply(lambda x: get_unique_value(x))

    def preprocess_text_fields(self):
        """
        Preprocess text fields in the text DataFrame by joining unique values.
        """
        self.df_text["pdt_inclexcl_ENG_CONTENT"] = [
            ". ".join(list(set([str(el) for el in x])))
            for x in self.df_text["pdt_inclexcl_ENG_CONTENT"]
        ]
        self.df_text["pdt_product_detail_PRODUCTDESCRIPTION"] = [
            ". ".join(list(set([str(el) for el in x])))
            for x in self.df_text["pdt_product_detail_PRODUCTDESCRIPTION"]
        ]
        self.df_text["pdt_product_detail_PRODUCTTITLE"] = [
            ". ".join(list(set([str(el) for el in x])))
            for x in self.df_text["pdt_product_detail_PRODUCTTITLE"]
        ]
        self.df_text["pdt_product_detail_TOURGRADEDESCRIPTION"] = [
            list(set([str(el) for el in x]))
            for x in self.df_text["pdt_product_detail_TOURGRADEDESCRIPTION"]
        ]

    def assert_sizes(self):
        """
        Assert that the sizes of the main DataFrame and the text DataFrame are equal.
        """
        assert list(self.df[self.key_field]) == list(self.df_text[self.key_field])

    def astypestr(self):
        """
        Convert all values in both DataFrames to strings and replace 'nan' with empty strings.
        """
        self.df = self.df.astype(str)
        self.df = self.df.replace("nan", "")
        self.df_text = self.df_text.astype(str)
        self.df_text = self.df_text.replace("nan", "")
