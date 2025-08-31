import argparse
import io
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import requests
from PIL import Image
import matplotlib.pyplot as plt


def _post_point(api_base: str, img_bytes: bytes, label: str) -> List[dict]:
    url = f"{api_base.rstrip('/')}/point"
    r = requests.post(
        url,
        data={"label": label},
        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return data["points"] if isinstance(data, dict) and "points" in data else data


def _to_xy(
    p: Dict[str, Union[int, float, Dict]],
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Return (x,y) in pixel coords. Supports normalized or absolute formats."""
    if "point" in p and isinstance(p["point"], dict):
        p = p["point"]

    # Accept common key variants
    x = p.get("x", p.get("cx", p.get("px", None)))
    y = p.get("y", p.get("cy", p.get("py", None)))
    if x is None or y is None:
        raise ValueError(f"Unsupported point format: {p}")

    x, y = float(x), float(y)
    W, H = image_size

    # Scale normalized coords
    if max(x, y) <= 1.0:
        x *= W
        y *= H

    # Clamp
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    return x, y


def show_with_points(
    image: Image.Image,
    points: Iterable[Dict],
    label: str,
):
    W, H = image.size
    xs, ys = [], []
    for p in points or []:
        try:
            x, y = _to_xy(p, (W, H))
            xs.append(x)
            ys.append(y)
        except Exception:
            continue

    plt.figure()
    plt.imshow(image)
    if xs:
        ms = max(4, int(min(W, H) * 0.01))  # marker size relative to image
        plt.scatter(
            xs, ys, marker="x", s=ms**2, linewidths=2, c="red"
        )  # markers are red
        # Optional labels near points
        for x, y in zip(xs, ys):
            plt.text(x + 30, y - 30, label, fontsize=9, color="red")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    """Use api running in the background and detect points with given label."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path, help="Path to input image")
    ap.add_argument("--label", required=True, help="Label to locate (e.g., 'person')")
    ap.add_argument("--api", default="http://localhost:9999", help="API base URL")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)

    points = _post_point(args.api, buf.getvalue(), args.label)
    show_with_points(img, points, args.label)


if __name__ == "__main__":
    """
    uv run demo_app.py \
        --image "path/to/image" \
        --label "object"
    """
    main()
