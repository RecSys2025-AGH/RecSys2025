import polars as pl
import data_actions.data_processing.utils as utils

def main():
    page_visit = pl.scan_parquet("data/original_data/page_visit.parquet")
    page_visit = utils.convert_timestamp_to_datetime(page_visit)
    min_day = page_visit.select("timestamp").min().collect().item()

    page_visit = page_visit.with_columns(
        (pl.col("timestamp") - min_day).dt.total_days().alias("days_since_start")
    )
    page_visits_agg = (
        page_visit
        .select(["client_id", "days_since_start", "url"])
        .group_by(["client_id", "days_since_start"])
        .agg([
            pl.count("client_id").alias("page_visits_count"),
            pl.col("url")
              .value_counts(sort=True)
              .struct.field("url")
              .first()
              .alias("most_common_visited_page")
        ])
    )

    dataset = pl.scan_parquet("data/processed_data/dataset.parquet")
    
    final_dataset = dataset.join(page_visits_agg, on=["client_id", "days_since_start"], how="full")
    
    final_dataset.sink_parquet("data/processed_data/dataset_with_urls.parquet")

if __name__ == "__main__":
    main()
