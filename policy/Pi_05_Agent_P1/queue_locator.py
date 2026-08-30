"""File-queue stand-in for the locator, for single-episode diagnosis.

Speaks the same HTTP interface as `locator_server.py`, but instead of running a
local model it writes each request to disk and blocks until an answer file
appears. That lets a stronger vision-language model answer the same questions
from the same head-camera image, which separates two failure modes that the
task score cannot distinguish: weak grounding versus weak manipulation.

This is a diagnostic tool, not an evaluation configuration. Answers arrive
out-of-band, so it cannot drive a full multi-layout sweep; use
`locator_server.py` for anything reported as a result.
"""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


class RequestQueue:
    def __init__(self, directory: str, timeout_s: float, poll_s: float = 1.0):
        self.directory = directory
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        os.makedirs(self.directory, exist_ok=True)
        self._index = 0
        self._lock = threading.Lock()

    def ask(self, image: Image.Image, kind: str, prompt: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            index = self._index
            self._index += 1

        stem = os.path.join(self.directory, f"{index:05d}_{kind}")
        image.save(f"{stem}.png")
        with open(f"{stem}.request.json", "w") as handle:
            json.dump({"kind": kind, **prompt}, handle, indent=2)
        print(f"[queue-locator] waiting on {stem}.response.json", flush=True)

        deadline = time.monotonic() + self.timeout_s
        response_path = f"{stem}.response.json"
        while time.monotonic() < deadline:
            if os.path.exists(response_path):
                # The writer may still be flushing, so tolerate a partial file.
                try:
                    with open(response_path) as handle:
                        return json.load(handle)
                except json.JSONDecodeError:
                    time.sleep(self.poll_s)
                    continue
            time.sleep(self.poll_s)
        raise TimeoutError(f"No answer appeared at {response_path}.")


class QueueLocatorHandler(BaseHTTPRequestHandler):
    queue: RequestQueue

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        if self.path not in {"/locate", "/locate_batch", "/locate_baskets"}:
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 64 * 1024 * 1024:
                raise ValueError(f"Invalid request size: {content_length}.")
            payload = json.loads(self.rfile.read(content_length))

            if self.path == "/locate":
                result = self.queue.ask(
                    self._decode_image(payload),
                    "object",
                    {"instruction": str(payload.get("instruction", ""))},
                )
            elif self.path == "/locate_baskets":
                result = self.queue.ask(self._decode_image(payload), "baskets", {})
            else:
                items = payload.get("items")
                if not isinstance(items, list) or not items:
                    raise ValueError("Batch request requires a non-empty items list.")
                result = {
                    "results": [
                        self.queue.ask(
                            self._decode_image(item),
                            "object",
                            {"instruction": str(item.get("instruction", ""))},
                        )
                        for item in items
                    ]
                }
            self._send_json(result)
        except Exception as exc:
            print(f"[queue-locator] request failed: {type(exc).__name__}: {exc}", flush=True)
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[queue-locator] {self.address_string()} {format % args}", flush=True)

    @staticmethod
    def _decode_image(payload: dict[str, Any]) -> Image.Image:
        shape = tuple(int(value) for value in payload["shape"])
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Expected HWC RGB shape, got {shape}.")
        raw = base64.b64decode(payload["rgb_u8"], validate=True)
        return Image.fromarray(np.ndarray(shape, dtype=np.uint8, buffer=raw), mode="RGB")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args()

    QueueLocatorHandler.queue = RequestQueue(args.queue_dir, args.timeout_s)
    server = ThreadingHTTPServer((args.host, args.port), QueueLocatorHandler)
    print(
        f"[queue-locator] ready at http://{args.host}:{args.port} "
        f"queue={args.queue_dir}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
