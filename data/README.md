# Data Directory

Place your NYC Taxi dataset CSV file(s) here.

## Dataset Information

### For Google Colab:
- Use a smaller subset: **5 million rows**
- File size should be manageable (< 2GB recommended)

### For Local Setup:
- Use the full dataset: **38 million rows**
- Full dataset provides better model performance

## Dataset File

**Default filename**: `Distilled_2023_Yellow_Taxi_Trip_Data.csv`

You can use either:
- **Smaller version**: 5 million rows (for Colab/testing)
- **Full version**: 38 million rows (for local setup)

## Expected Columns

The dataset should include the following columns (NYC Taxi standard format):

- `tpep_pickup_datetime` - Pickup timestamp (format: "%m/%d/%Y %I:%M:%S %p")
- `tpep_dropoff_datetime` - Dropoff timestamp (format: "%m/%d/%Y %I:%M:%S %p")
- `pickup_latitude` - Pickup latitude
- `pickup_longitude` - Pickup longitude
- `dropoff_latitude` - Dropoff latitude
- `dropoff_longitude` - Dropoff longitude
- `trip_distance` - Trip distance in miles
- `fare_amount` - Fare amount
- `tip_amount` - Tip amount (target variable)
- `passenger_count` - Number of passengers
- `vendor_id` - Vendor ID
- `payment_type` - Payment type
- `tolls_amount` - Tolls amount
- `total_amount` - Total amount
- `mta_tax` - MTA tax
- `extra` - Extra charges
- `store_and_fwd_flag` - Store and forward flag (will be dropped - not meaningful for tips)

## Download Links

- [NYC Taxi Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Search for "Yellow Taxi Trip Records"

## Notes

- Update `DATASET_FILE` in `config.py` to match your filename
- The script will automatically handle missing columns
- Ensure the CSV file is properly formatted
