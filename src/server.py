import requests
import sys
import time
import random
import json
import string
import urllib.parse

from flask import Flask, Response, Request, request

from ai import AI
from settings import (
    DIR_MODEL,
    PATH_OBJECTS_YAML,
    PATH_RAINBOW_YAML,
)

class Server:
    def __init__(self, host, port, ai):
        self.api = api = Flask(__name__)
        self.host = host
        self.port = port
        self.ai = ai

        def send_json(json):
            resp = Response(json)
            resp.headers.set("content-type", "application/json")
            return resp

        def random_str(length):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(length))

        @api.route("/recognition", methods=["GET"])
        def recognize():
            params = request.args

            if not "url" in params:
                return send_json(json.dumps({ "error": "No url provided" }))

            if not "label" in params:
                return send_json(json.dumps({ "error": "No label provided" }))

            url = urllib.parse.unquote(params["url"])
            label = urllib.parse.unquote(params["label"])

            st = time.time()
            img_bytes = requests.get(url).content
            res = ai.predict(img_bytes, label)
            et = time.time() - st

            return send_json(json.dumps({ "result": res, "time": str(et)}))

    def start(self):
        self.api.run(host=self.host, port=self.port, debug=False)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8080

    for arg in sys.argv:
        if "=" in arg:
            split = arg.split("=")
            key = split[0]
            value = split[1]

            if key == "--host":
                host = value
            elif key == "--port":
                port = int(value)

    # Updating objects.yaml file
    objects = requests.get("https://raw.githubusercontent.com/QIN2DIM/hcaptcha-challenger/main/src/objects.yaml").content
    with open("objects.yaml", "wb") as writer:
        writer.write(objects)

    ai = AI(
        dir_model=DIR_MODEL,
        onnx_prefix=None,
        path_objects_yaml=PATH_OBJECTS_YAML,
        path_rainbow_yaml=PATH_RAINBOW_YAML,
    )

    server = Server(host, port, ai)
    server.start()

