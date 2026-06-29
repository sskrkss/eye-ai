import base64
import json
from uuid import UUID

import pika

from database.config import get_settings

setting = get_settings()
connection_params = pika.ConnectionParameters(
    host=setting.RMQ_HOST,
    port=setting.RMQ_PORT,
    virtual_host='/',
    credentials=pika.PlainCredentials(
        username=setting.RMQ_USER,
        password=setting.RMQ_PASS
    ),
    heartbeat=30,
    blocked_connection_timeout=2
)


def send_task(task_id: UUID, image_bytes: bytes):
    message = {
        "task_id": str(task_id),
        "features": {
            "image_base64": base64.b64encode(image_bytes).decode("utf-8")
        },
        "model": "dr_detection"
    }

    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()
    channel.queue_declare(queue=setting.RMQ_QUEUE)
    channel.basic_publish(
        exchange='',
        routing_key=setting.RMQ_QUEUE,
        body=json.dumps(message).encode('utf-8')
    )
    connection.close()
