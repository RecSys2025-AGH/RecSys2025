import polars as pl
import numpy as np
import data_actions.data_processing.utils as utils

class DailyDataset:
    """
    Class to create a daily dataset from the original data.
    """
    def __init__(self, data_path: str, output_path: str) -> None:
        """Initialized the class with the data path and output path.
        It loads the data and initializes the event counts.

        Args:
            data_path (str): path to the original data
            output_path (str): path where the data will be saved
        """
        self.output_path = output_path
        self.data = self.load_data(data_path)
        self.event_counts = dict()
    
    def load_data(self, data_path: str) -> dict:
        """Load the data from the data path and calculate the days since start.

        Args:
            data_path (str): path to the original data

        Returns:
            dict: dictionary with the data, with the same structure as utils.load_data
        """
        data = utils.load_data(data_path)
        utils.calculate_days_since_start(data)
        return data
    
    def create_dataset(self) -> None:
        """Creates the dataset
        """
        self.add_product_properties()
        self.count_events_per_day()
        self.count_page_visits()
        self.count_queries()
        dataset = self.join_tables()
        self.save_dataset(dataset)


    def save_dataset(self, dataset: pl.LazyFrame):
        """Save the dataset to the output path.

        Args:
            dataset (pl.LazyFrame): dataset to save
        """
        dataset.sink_parquet(self.output_path + "/dataset.parquet")

    def add_product_properties(self):
        """ Add product properties to the events tables.
        The product properties are added to the events tables by joining the product properties table with each event table.
        """
        product_info = self.data["products"]["product_properties"]
        for table in ("add_to_cart", "remove_from_cart", "product_buy"):
            df = self.data["events"][table]
            df = df.join(product_info.select(pl.exclude("name")), on="sku", how="left")
            self.data["events"][table] = df
    
    def count_events_per_day(self):
        """ Count the events per day for each client.
        The events are counted by grouping by client_id and days_since_start, and counting the number of events.
        """
        for table in ("add_to_cart", "remove_from_cart", "product_buy"):
            df = self.data["events"][table]
            self.event_counts[table] = df.group_by(["client_id", "days_since_start"]).agg(
                [
                    pl.col("client_id").count().alias(f"count_{table}"),
                    pl.col("sku").value_counts(sort=True).struct.field("sku").first().alias(f"most_common_item_{table}"),
                    pl.col("category").value_counts(sort=True).struct.field("category").first().alias(f"most_common_cat_{table}"),
                    pl.col("price").mean().alias(f"avg_price_{table}"),
                    pl.col("price").value_counts(sort=True).struct.field("price").first().alias(f"most_common_price_{table}"),
                ]
            )
    
    def count_page_visits(self):
        """ Count the page visits per day for each client.
        The page visits are counted by grouping by client_id and days_since_start, and counting the number of visits.
        """
        page_visits = self.data["events"]["page_visit"].select("client_id", "days_since_start", "url").group_by("client_id", "days_since_start").agg(
        [
            pl.col("client_id").count().alias("page_visits_count"),
            #pl.col("url").value_counts(sort=True).struct.field("url").first().alias(f"most_common_visited_page")
        ])
        self.event_counts["page_visit"] = page_visits
    
    def count_queries(self):
        """ Count the search queries per day for each client.
        The search queries are counted by grouping by client_id and days_since_start, and counting the number of queries.
        """
        query_counts = self.data["events"]["search_query"].select("client_id", "days_since_start").group_by("client_id", "days_since_start").count()
        query_counts = query_counts.rename({"count": "search_query_count"})
        self.event_counts["search_query"] = query_counts

    def join_tables(self):
        """ Join the tables together.
        The tables are joined by client_id and days_since_start, and the missing values are filled with 0.
        """
        joined_table = list(self.event_counts.items())[0][1]
        for _, table in list(self.event_counts.items())[1:]:
            joined_table = joined_table.join(table, on=["client_id", "days_since_start"], how="full", coalesce=True)

        joined_table = joined_table.fill_null(0)
        return joined_table

if __name__ == "__main__":
    data_path = "data/original_data"
    output_path = "data/processed_data"
    dataset = DailyDataset(data_path, output_path)
    dataset.create_dataset()