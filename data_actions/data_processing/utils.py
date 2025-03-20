import polars as pl
import numpy as np

def load_data(data_folder):
    """Load the RecSys dataset.

    Args:
        data_folder (string): relative path to data folder where the data is stored

    Returns:
        dict: dictionary containing the data with structure:
            {
                "events": {"add_to_cart", "page_visit", "product_buy", "remove_from_cart", "search_query"},
                "products": {"product_properties"},
                "inputs": {"relevant_clients"},
                "targets": {"active_clients", "popularity_propensity_category", "popularity_propensity_sku", "propensity_category", "propensity_sku"}
        }
    """
    events = {
        "add_to_cart": pl.scan_parquet(f"{data_folder}/add_to_cart.parquet"),
        "page_visit": pl.scan_parquet(f"{data_folder}/page_visit.parquet"),
        "product_buy": pl.scan_parquet(f"{data_folder}/product_buy.parquet"),
        "remove_from_cart": pl.scan_parquet(f"{data_folder}/remove_from_cart.parquet"),
        "search_query": pl.scan_parquet(f"{data_folder}/search_query.parquet"),
    }

    products = {
        "product_properties": pl.scan_parquet(f"{data_folder}/product_properties.parquet")
    }

    inputs = {
        "relevant_clients": np.load(f"{data_folder}/input/relevant_clients.npy", mmap_mode="r")
    }

    targets = {
        "active_clients": np.load(f"{data_folder}/target/active_clients.npy", mmap_mode="r"),
        "popularity_propensity_category": np.load(f"{data_folder}/target/popularity_propensity_category.npy", mmap_mode="r"),
        "popularity_propensity_sku": np.load(f"{data_folder}/target/popularity_propensity_sku.npy", mmap_mode="r"),
        "propensity_category": np.load(f"{data_folder}/target/propensity_category.npy", mmap_mode="r"),
        "propensity_sku": np.load(f"{data_folder}/target/propensity_sku.npy", mmap_mode="r"),
    }

    data = {
        "events": events,
        "products": products,
        "inputs": inputs,
        "targets": targets
    }

    return data




def convert_timestamp_to_datetime(df):
    """Converts timestamp column to datetime.

    Args:
        df (pl.LazyFrame): Table from the dataset

    Returns:
        pl.LazyFrame: Table with column timestamp changed type to datetime
    """
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S")
        )
    return df