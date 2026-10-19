"""
Thin wrapper around paho-mqtt for publishing JSON payloads.
Imported by sensor_simulator.py (and anything else that needs to publish).
"""

import json
import paho.mqtt.client as mqtt
from config import MQTT_BROKER, MQTT_PORT, MQTT_CLIENT_ID_PUBLISHER


class MQTTPublisher:
    def __init__(self, client_id=MQTT_CLIENT_ID_PUBLISHER, broker=MQTT_BROKER, port=MQTT_PORT):
        self.client = mqtt.Client(client_id=client_id)
        self.broker = broker
        self.port = port
        self._connected = False

    def connect(self):
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        self._connected = True
        print(f"[publisher] connected to {self.broker}:{self.port}")

    def publish(self, topic, payload: dict, qos=1):
        if not self._connected:
            self.connect()
        message = json.dumps(payload)
        info = self.client.publish(topic, message, qos=qos)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"MQTT publish failed for {topic!r} (error code {info.rc})"
            )
        info.wait_for_publish(timeout=30)
        if not info.is_published():
            raise TimeoutError(f"MQTT publish timed out for {topic!r}")

    def disconnect(self):
        if self._connected:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False
