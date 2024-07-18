#!/usr/bin/env python
# coding: utf-8

import gc
import re
from collections import OrderedDict

from google.cloud import bigquery
from pandas_gbq import to_gbq

# Run garbage collection to free up memory.
gc.collect()


class BigQueryDataProcessor:
    def __init__(
        self, config, dataset_id, table_id, table_fields, key_field, time_feature
    ):
        """
        Initialize the BigQueryDataProcessor with the given parameters.

        Parameters:
        config (str): The configuration dictionary.
        dataset_id (str): The dataset ID in BigQuery.
        table_id (str): The table ID in BigQuery.
        table_fields (list of str): The fields to be collected.
        key_field (str): The column name to group by.
        time_feature (bool): Flag indicating if time-based aggregation is needed.
        """
        self.config = config
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_fields = table_fields
        self.selected_fields = ", ".join(self.table_fields)
        self.key_field = key_field
        self.time_feature = time_feature

        self.client = bigquery.Client()

    def read_bigquery_table(self):
        """
        Read a BigQuery table into a DataFrame and aggregate its values.

        This method reads data from the specified BigQuery table, cleans up the column names,
        and calls the aggregation method to process the DataFrame.

        Returns:
        None
        """
        table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
        table = self.client.get_table(table_ref)

        if self.time_feature:
            query = f"SELECT ProductCode, MAX(OrderDate) AS MostRecentOrderDate FROM `{table.project}.{table.dataset_id}.{table.table_id}` WHERE OrderDate IS NOT NULL GROUP BY ProductCode"
        else:
            query = f"SELECT {self.selected_fields} FROM `{table.project}.{table.dataset_id}.{table.table_id}`"

        query_job = self.client.query(query)
        self.df = query_job.to_dataframe()

        prefix = self.table_id + "_"
        self.df.columns = [
            prefix + re.sub(r"[^A-Za-z0-9]+", "", col.upper())
            for col in self.df.columns
        ]
        self.df.rename(columns={prefix + self.key_field: self.key_field}, inplace=True)

        self.aggregate_and_get_set()

    def aggregate_and_get_set(self):
        """
        Aggregate and deduplicate values in each column of the DataFrame based on the key field.

        This method groups the DataFrame by the key field and aggregates the other columns,
        removing duplicate values in each list.

        Returns:
        None
        """
        columns_to_aggregate = [col for col in self.df.columns if col != self.key_field]
        agg_dict = {col: list for col in columns_to_aggregate}
        self.df = self.df.groupby(self.key_field).agg(agg_dict).reset_index()

        for column in columns_to_aggregate:
            self.df[column] = [
                list(OrderedDict.fromkeys(el).keys()) for el in self.df[column]
            ]

    def write_to_bigquery(self, destination_table_id, project_id, if_exists="replace"):
        """
        Write the DataFrame to a BigQuery table.

        Parameters:
        destination_table_id (str): The destination table ID in BigQuery.
        project_id (str): The Google Cloud project ID.
        if_exists (str): What to do if the table exists. Default is 'replace'.

        Returns:
        None
        """
        table_id = f"{self.dataset_id}.{destination_table_id}"
        to_gbq(self.df, table_id, project_id=project_id, if_exists=if_exists)
