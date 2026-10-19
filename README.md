# P_220 — AI-Based Industrial Process Quality Prediction

Embedded IoT · MQTT + Python + XGBoost + Streamlit

## Files
- `config.py` — all settings (broker, topics, sensor ranges, file paths)
- `mqtt_publisher.py` — reusable MQTT publish helper
- `sensor_simulator.py` — replays complete rows from the mining dataset over MQTT
- `train_ml_model.py` — trains the XGBoost model, saves it to `models/ml_model.joblib`
- `mqtt_subscriber.py` — listens for readings, predicts, logs, writes `latest_data.json`
- `dashboard.py` — Streamlit live view
- `latest_data.json` — starter file so the dashboard has something to show before the pipeline runs
- `data/` — training data + prediction log (created automatically)
- `models/` — saved model (created automatically)

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Also install and run a local MQTT broker, e.g. [Mosquitto](https://mosquitto.org/download/).

## Run order
Open four terminals (activate the venv in each):

1. Start Mosquitto (or `mosquitto` as a service).
2. `python train_ml_model.py` — evaluates on a chronological holdout, then fits
   and saves the final model using all usable mining-dataset rows.
3. `python mqtt_subscriber.py` — subscribes and predicts silica concentrate.
4. `python sensor_simulator.py` — replays every complete mining-dataset row.
5. `streamlit run dashboard.py` — opens the live dashboard in your browser.

The simulator runs once through the dataset by default and publishes with
MQTT QoS 1. Replay as fast as the broker accepts messages, or test a subset:

```bash
python sensor_simulator.py --limit 100
python sensor_simulator.py --start-row 1000
python sensor_simulator.py --interval 1
```

The payload contains all 21 model input features, the source date, and the
actual concentrate measurements. The subscriber uses only the model features
for its prediction; actual measurements are retained for dashboard comparison.
The subscriber uses a persistent MQTT session and subscribes at QoS 1, so start
it before replaying the dataset.
