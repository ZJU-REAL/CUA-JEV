"""Synthetic-image gateway probe; never uploads a desktop screenshot."""

from __future__ import annotations

import argparse
import io
import json
import os

from PIL import Image, ImageDraw

from cua_jev.model_planner import ChatModelPlanner
from cua_jev.vision import WindowImage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-insecure-http", action="store_true")
    args = parser.parse_args()
    canvas = Image.new("RGB", (400, 240), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 28), "Settings", fill="#172536")
    draw.rectangle((264, 167, 372, 218), fill="#145ea8")
    draw.text((298, 185), "Save", fill="white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=75)
    sample = canvas.convert("L").resize((64, 64)).tobytes()
    planner = ChatModelPlanner(
        args.base_url, args.model,
        api_key=os.getenv("CUA_JEV_MODEL_API_KEY"),
        allow_insecure_http=args.allow_insecure_http,
        use_env_proxy=False,
    )
    try:
        scene = planner.perceive_scene(
            "Find the Save button", "Synthetic Settings", "", WindowImage(
                buffer.getvalue(), sample, (0, 0, 400, 240)
            )
        )
        print(json.dumps({
            "scene": scene.to_dict(), "usage": planner.last_usage,
            "vision_wall_ms": round(planner.vision_wall_ms, 1),
        }, indent=2))
    finally:
        planner.close()


if __name__ == "__main__":
    main()
