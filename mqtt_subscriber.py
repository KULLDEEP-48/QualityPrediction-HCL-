"""
Subscribes to the sensor topic, runs each reading through the trained
model, writes the latest prediction to LATEST_DATA_PATH (for the dashboard)
and appends every prediction to PREDICTIONS_LOG_PATH (for history/evaluation).

Run train_ml_model.py at least once before starting this.

Usage:
    python mqtt_subscriber.py
"""

import json
import os
import csv
import math
import tempfile
import time
from datetime import datetime, timezone

import joblib
import pandas as pd
import paho.mqtt.client as mqtt
from xgboost import XGBRegressor

from config import (
    MQTT_BROKER,
    MQTT_PORT,
    MQTT_TOPIC_SENSORS,
    MQTT_CLIENT_ID_SUBSCRIBER,
    MODEL_PATH,
    LATEST_DATA_PATH,
    PREDICTIONS_LOG_PATH,
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run train_ml_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    if (
        not isinstance(model, XGBRegressor)
        or not hasattr(model, "feature_names_in_")
        or getattr(model, "target_name_", None) != "% Silica Concentrate"
    ):
        raise ValueError(
            f"Model at {MODEL_PATH} is not a compatible silica-concentrate "
            "regressor. "
            "Retrain it with train_ml_model.py."
        )
    return model


def log_prediction(row: dict):
    log_dir = os.path.dirname(PREDICTIONS_LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    fieldnames = list(row)
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    temporary_path = None
    try:
        with open(PREDICTIONS_LOG_PATH, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            old_fieldnames = reader.fieldnames or []
            if old_fieldnames == fieldnames:
                with open(
                    PREDICTIONS_LOG_PATH, "a", newline="", encoding="utf-8"
                ) as output:
                    csv.DictWriter(output, fieldnames=fieldnames).writerow(row)
                return

            merged_fieldnames = list(dict.fromkeys([*old_fieldnames, *fieldnames]))
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=log_dir or ".",
                prefix="predictions_log_",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = temporary.name
                writer = csv.DictWriter(temporary, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(reader)
                writer.writerow(row)

        for attempt in range(5):
            try:
                os.replace(temporary_path, PREDICTIONS_LOG_PATH)
                break
            except PermissionError as error:
                if attempt == 4:
                    raise PermissionError(
                        f"Cannot update {PREDICTIONS_LOG_PATH}; close it in "
                        "Excel or another program and pause OneDrive syncing, "
                        "then restart the subscriber."
                    ) from error
                time.sleep(0.2 * (attempt + 1))
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


def write_latest(row: dict):
    with open(LATEST_DATA_PATH, "w") as f:
        json.dump(row, f, indent=2)


def make_on_message(model):
    feature_columns = list(model.feature_names_in_)
    target_name = model.target_name_

    def on_message(_client, _userdata, msg):
        try:
            reading = json.loads(msg.payload.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            print("[subscriber] payload is not valid UTF-8 JSON, skipping")
            return

        if not isinstance(reading, dict):
            print("[subscriber] payload must be a JSON object, skipping")
            return

        missing_features = [name for name in feature_columns if name not in reading]
        if missing_features:
            print(
                "[subscriber] payload is missing mining process features: "
                + ", ".join(missing_features)
            )
            return

        try:
            feature_values = {
                name: float(reading[name]) for name in feature_columns
            }
        except (TypeError, ValueError):
            print("[subscriber] mining process features must be numeric, skipping")
            return
        if not all(math.isfinite(value) for value in feature_values.values()):
            print("[subscriber] mining process features must be finite, skipping")
            return

        features = pd.DataFrame([feature_values], columns=feature_columns)
        prediction = float(model.predict(features)[0])

        result = {
            **reading,
            f"predicted_{target_name}": round(prediction, 4),
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        write_latest(result)
        log_prediction(result)
        print(
            f"[subscriber] batch {reading.get('batch_id')} -> "
            f"predicted {target_name}={prediction:.4f}"
        )

    return on_message


def main():
    model = load_model()

    client = mqtt.Client(
        client_id=MQTT_CLIENT_ID_SUBSCRIBER,
        clean_session=False,
    )
    client.on_message = make_on_message(model)

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.subscribe(MQTT_TOPIC_SENSORS, qos=1)
    print(f"[subscriber] listening on {MQTT_TOPIC_SENSORS}. Ctrl+C to stop.")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[subscriber] stopped.")


if __name__ == "__main__":
    main()
