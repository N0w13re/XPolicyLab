"""Small localhost service for language-conditioned 2D object grounding."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SYSTEM_PROMPT = """You ground robot-manipulation targets in a head-camera image.
The instruction asks to sort objects into three baskets: left, middle, and
right, each for a different object category. Read the full instruction and the
current scene. Find the NEXT movable object that should be picked from the
table now. Ignore baskets, trays, robot arms, and objects already inside their
destination container. When several categories still have objects on the table,
prioritize in this order: left-basket category, then middle-basket category,
then right-basket category. If several matching objects remain on the table,
choose the clearest reachable one. The "label" must be copied verbatim from one
of the object categories named in the instruction, because it decides which
basket the object is later placed in. Return one JSON object only:
{"label": "object category", "bbox_2d": [x0, y0, x1, y1]}
Coordinates are relative integers from 0 to 1000. Return
{"label": null, "bbox_2d": null} if no table object still needs manipulation."""

HELD_PROMPT = """You identify the object a robot gripper is currently holding.
The image is from a wrist camera, so the held object is the one in the gripper
jaws, near the image centre and close to the lens. Ignore the table, baskets and
other objects further away. Answer with the object category from the instruction
that the held object belongs to, copied verbatim. Return one JSON object only:
{"label": "object category"}
Return {"label": null} if the gripper is empty or the object is unclear."""

BASKET_PROMPT = """You locate the three sorting baskets in a robot head-camera image.
The table holds exactly three baskets in a row. Report a bounding box for each,
ordered by horizontal position in the image: the leftmost basket is "left", the
centre one is "middle", the rightmost is "right". Return one JSON object only:
{"left": [x0, y0, x1, y1], "middle": [x0, y0, x1, y1], "right": [x0, y0, x1, y1]}
Coordinates are relative integers from 0 to 1000."""


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Model did not return JSON: {text!r}")
    result = json.loads(match.group(0))
    bbox = result.get("bbox_2d")
    if bbox is None and "bbox_2d" in result:
        # An explicit null is the documented answer for "the table is clear",
        # which the caller needs in order to finish the instruction.
        return {"label": result.get("label"), "bbox_2d": None}
    if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Model did not return one bounding box: {result!r}")
    result["bbox_2d"] = [float(value) for value in bbox]
    return result


def _extract_basket_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Model did not return JSON: {text!r}")
    parsed = json.loads(match.group(0))
    result: dict[str, Any] = {}
    for side in ("left", "middle", "right"):
        bbox = parsed.get(side)
        if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Missing bounding box for the {side} basket: {parsed!r}")
        result[side] = [float(value) for value in bbox]
    return result


class QwenObjectLocator:
    def __init__(self, model_path: str, device: str):
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        # Decoder-only batched generation must left-pad so every sample's
        # continuation starts after its own complete multimodal prompt.
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    def locate(self, image: Image.Image, instruction: str) -> dict[str, Any]:
        return self.locate_batch([image], [instruction])[0]

    def classify_held(self, image: Image.Image, instruction: str) -> dict[str, Any]:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": HELD_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Robot instruction: {instruction}"},
                ],
            },
        ]
        text = self._generate([conversation])[0]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError(f"Model did not return JSON: {text!r}")
        parsed = json.loads(match.group(0))
        return {"label": parsed.get("label"), "raw_response": text}

    def locate_baskets(self, image: Image.Image) -> dict[str, Any]:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": BASKET_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Locate the three baskets."},
                ],
            },
        ]
        text = self._generate([conversation])[0]
        result = _extract_basket_json(text)
        result["raw_response"] = text
        return result

    def locate_batch(
        self,
        images: list[Image.Image],
        instructions: list[str],
    ) -> list[dict[str, Any]]:
        if not images or len(images) != len(instructions):
            raise ValueError("images and instructions must have the same non-zero length.")
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Robot instruction: {instruction}"},
                    ],
                },
            ]
            for image, instruction in zip(images, instructions)
        ]
        texts = self._generate(conversations)
        results = []
        for text in texts:
            try:
                result = _extract_json(text)
                result["raw_response"] = text
            except Exception as exc:
                result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_response": text,
                }
            results.append(result)
        return results

    def _generate(self, conversations: list[list[dict[str, Any]]]) -> list[str]:
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        with self._lock, torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class LocatorRequestHandler(BaseHTTPRequestHandler):
    locator: QwenObjectLocator

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        if self.path not in {
            "/locate",
            "/locate_batch",
            "/locate_baskets",
            "/classify_held",
        }:
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 8 * 1024 * 1024:
                raise ValueError(f"Invalid request size: {content_length}.")
            payload = json.loads(self.rfile.read(content_length))
            if self.path == "/locate":
                image = self._decode_image(payload)
                result = self.locator.locate(image, str(payload["instruction"]))
            elif self.path == "/locate_baskets":
                result = self.locator.locate_baskets(self._decode_image(payload))
            elif self.path == "/classify_held":
                result = self.locator.classify_held(
                    self._decode_image(payload),
                    str(payload.get("instruction", "")),
                )
            else:
                items = payload.get("items")
                if not isinstance(items, list) or not items:
                    raise ValueError("Batch request requires a non-empty items list.")
                images = [self._decode_image(item) for item in items]
                instructions = [str(item["instruction"]) for item in items]
                result = {"results": self.locator.locate_batch(images, instructions)}
            self._send_json(result)
        except Exception as exc:
            print(f"[locator] request failed: {type(exc).__name__}: {exc}", flush=True)
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[locator] {self.address_string()} {format % args}", flush=True)

    @staticmethod
    def _decode_image(payload: dict[str, Any]) -> Image.Image:
        shape = tuple(int(value) for value in payload["shape"])
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"Expected HWC RGB shape, got {shape}.")
        raw = base64.b64decode(payload["rgb_u8"], validate=True)
        image_array = np.ndarray(shape, dtype=np.uint8, buffer=raw)
        return Image.fromarray(image_array, mode="RGB")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    LocatorRequestHandler.locator = QwenObjectLocator(args.model_path, args.device)
    server = ThreadingHTTPServer((args.host, args.port), LocatorRequestHandler)
    print(f"[locator] ready at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
