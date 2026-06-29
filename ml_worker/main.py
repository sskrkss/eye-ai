import base64
import json
import os
import socket

import pika
import requests
from jose import jwt

from ml_model import MlModel


connection_params = pika.ConnectionParameters(
    host=os.getenv("RMQ_HOST"),
    port=int(os.getenv("RMQ_PORT")),
    virtual_host="/",
    credentials=pika.PlainCredentials(
        username=os.getenv("RMQ_USER"),
        password=os.getenv("RMQ_PASS")
    ),
    heartbeat=30,
    blocked_connection_timeout=2
)

connection = pika.BlockingConnection(connection_params)
channel = connection.channel()
queue_name = os.getenv("RMQ_QUEUE")
channel.queue_declare(queue=queue_name)

ml_model = MlModel()


def callback(ch, method, properties, body):
    message = json.loads(body.decode("utf-8"))

    try:
        image_bytes = base64.b64decode(message["features"]["image_base64"])
        prediction = ml_model.predict(image_bytes)
        task_status = "completed"
    except Exception:
        prediction = None
        task_status = "failed"

    s2s_token = jwt.encode(
        claims={"iss": "ml_worker", "sub": "app"},
        key=os.getenv("S2S_SECRET_KEY"),
        algorithm="HS256"
    )

    requests.post(
        f"{os.getenv('APP_URL')}/api/ml-tasks/save-prediction",
        json={
            "task_id": message["task_id"],
            "task_status": task_status,
            "prediction": prediction,
            "worker_id": socket.gethostname()
        },
        headers={"Authorization": f"Bearer {s2s_token}"},
        timeout=30
    )

    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_consume(
    queue=queue_name,
    on_message_callback=callback,
    auto_ack=False
)

channel.start_consuming()
