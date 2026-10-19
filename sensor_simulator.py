"""
Replays rows from the mining process dataset to MQTT in chronological order.

Usage:
    python sensor_simulator.py
    python sensor_simulator.py --interval 0 --limit 100
"""

import argparse
import math
import time

from config import MINING_DATA_PATH, MQTT_TOPIC_SENSORS, PUBLISH_INTERVAL_SECONDS
from mqtt_publisher import MQTTPublisher
from train_ml_model import DATE_COLUMN, OUTPUT_COLUMNS, load_dataset


def iter_readings(df, start_row=0, limit=None):
    """Yield complete process measurements, plus source date and actual targets."""
    feature_columns = [
        column
        for column in df.columns
        if column not in {DATE_COLUMN, *OUTPUT_COLUMNS}
    ]
    required_columns = [DATE_COLUMN, *feature_columns, *OUTPUT_COLUMNS]
    column_positions = {name: df.columns.get_loc(name) for name in required_columns}

    yielded = 0
    for row_number, values in enumerate(df.itertuples(index=False, name=None)):
        if row_number < start_row:
            continue
        if limit is not None and yielded >= limit:
            break

        reading = {
            name: float(values[column_positions[name]])
            for name in feature_columns
        }
        reading[DATE_COLUMN] = values[column_positions[DATE_COLUMN]].isoformat()
        reading["batch_id"] = row_number + 1
        for target in OUTPUT_COLUMNS:
            reading[target] = float(values[column_positions[target]])
        yield reading
        yielded += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=float,
        default=PUBLISH_INTERVAL_SECONDS,
        help="seconds between rows (default: replay as fast as MQTT acknowledges)",
    )
    parser.add_argument(
        "--start-row", type=int, default=0, help="zero-based dataset row to start at"
    )
    parser.add_argument(
        "--limit", type=int, help="maximum number of rows to replay (default: all)"
    )
    args = parser.parse_args()

    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--interval must be a finite number zero or greater")
    if args.start_row < 0:
        parser.error("--start-row must be zero or greater")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    df = load_dataset()
    if args.start_row >= len(df):
        parser.error(
            f"--start-row must be less than the {len(df)} available dataset rows"
        )
    total_rows = min(
        len(df) - args.start_row,
        args.limit if args.limit is not None else len(df) - args.start_row,
    )
    publisher = MQTTPublisher()
    publisher.connect()
    print(
        f"[simulator] replaying {total_rows:,} rows from {MINING_DATA_PATH}; "
        "publishes use MQTT QoS 1."
    )
    try:
        for sent, reading in enumerate(
            iter_readings(df, args.start_row, args.limit), start=1
        ):
            publisher.publish(MQTT_TOPIC_SENSORS, reading, qos=1)
            if sent == 1 or sent % 100 == 0 or sent == total_rows:
                print(
                    f"[simulator] acknowledged {sent:,}/{total_rows:,} rows "
                    f"(dataset batch {reading['batch_id']})"
                )
            if args.interval and sent < total_rows:
                time.sleep(args.interval)
    finally:
        publisher.disconnect()
    print("[simulator] dataset replay complete.")


if __name__ == "__main__":
    main()
