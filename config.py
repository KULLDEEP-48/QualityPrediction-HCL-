"""
Central configuration for the AI-Based Industrial Process Quality Prediction
system (P_220). Edit values here instead of hunting through each script.
"""

# --- MQTT Broker ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_SENSORS = "factory/line1/sensors"
MQTT_TOPIC_ALERTS = "factory/line1/alerts"
MQTT_CLIENT_ID_PUBLISHER = "quality_sensor_simulator"
MQTT_CLIENT_ID_SUBSCRIBER = "quality_predictor"

# --- Simulation ---
PUBLISH_INTERVAL_SECONDS = 0

# --- Model / files ---
MODEL_PATH = "models/ml_model.joblib"
MINING_DATA_PATH = "data/MiningProcess_Flotation_Plant_Database.csv"
LATEST_DATA_PATH = "latest_data.json"
PREDICTIONS_LOG_PATH = "data/predictions_log.csv"
