import threading
import textwrap
from typing import Dict, Iterable, List, Tuple, Union, Optional

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk
import tkinter as tk


# ----------------------
# server interaction
# ----------------------
def _post_point(api_base: str, img_bytes: bytes, label: str) -> List[dict]:
    url = f"{api_base.rstrip('/')}/point"
    r = requests.post(
        url,
        data={"label": label},
        files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data["points"] if isinstance(data, dict) and "points" in data else data


def _post_caption(api_base: str, img_bytes: bytes, length: str) -> str:
    url = f"{api_base.rstrip('/')}/caption"
    r = requests.post(
        url,
        data={"length": length},
        files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("caption", "")


def _to_xy(
    p: Dict[str, Union[int, float, Dict]],
    image_size: Tuple[int, int],
) -> Tuple[float, float]:
    if "point" in p and isinstance(p["point"], dict):
        p = p["point"]
    x = p.get("x", p.get("cx", p.get("px", None)))
    y = p.get("y", p.get("cy", p.get("py", None)))
    if x is None or y is None:
        raise ValueError(f"Unsupported point format: {p}")
    x, y = float(x), float(y)
    W, H = image_size
    if max(x, y) <= 1.0:  # normalized -> pixels
        x *= W
        y *= H
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    return x, y


# ----------------------
# app
# ----------------------
class CameraPointApp:
    def __init__(self, master: tk.Tk, cam_index: int = 0):
        self.master = master
        self.master.title("Point/Caption Client")

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # UI
        top = tk.Frame(master)
        top.pack(fill="x")

        tk.Label(top, text="API:").pack(side="left")
        self.api_var = tk.StringVar(value="http://localhost:9999")
        self.api_entry = tk.Entry(top, textvariable=self.api_var, width=28)
        self.api_entry.pack(side="left", padx=5)

        tk.Label(top, text="Feature:").pack(side="left")
        self.endpoint_var = tk.StringVar(value="point")
        self.endpoint_menu = tk.OptionMenu(
            top, self.endpoint_var, "point", "caption", command=self._on_endpoint_change
        )
        self.endpoint_menu.pack(side="left", padx=5)

        # Label input (for /point)
        tk.Label(top, text="Label:").pack(side="left")
        self.label_var = tk.StringVar(value="human")
        self.label_entry = tk.Entry(top, textvariable=self.label_var, width=14)
        self.label_entry.pack(side="left", padx=5)

        # Length selector (for /caption)
        tk.Label(top, text="Len:").pack(side="left")
        self.length_var = tk.StringVar(value="short")
        self.length_menu = tk.OptionMenu(top, self.length_var, "short", "normal")
        self.length_menu.pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(top, textvariable=self.status_var).pack(side="right", padx=8)

        self.canvas = tk.Label(master)
        self.canvas.pack()

        # state
        self.in_flight = False  # ensure only one request at a time
        self.last_points: Optional[List[dict]] = None  # last /point result
        self.last_caption: Optional[str] = None  # last /caption result
        self.last_frame_size: Tuple[int, int] = (1, 1)  # updated each draw
        self.running = True

        # initial enable/disable based on default endpoint
        self._on_endpoint_change(self.endpoint_var.get())

        # render loop
        self._schedule_render()

        # kick off first send immediately
        self._try_send_frame(initial=True)

        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- UI helpers
    def _on_endpoint_change(self, value: str):
        """Enable/disable config fields depending on endpoint."""
        if value == "point":
            self.label_entry.config(state="normal")
            self.length_menu.config(state="disabled")
        else:  # caption
            self.label_entry.config(state="disabled")
            self.length_menu.config(state="normal")

    def _set_status(self, text: str):
        self.master.after(0, self.status_var.set, text)

    # --- main loop
    def _schedule_render(self):
        if not self.running:
            return
        self._render()
        self.master.after(16, self._schedule_render)  # ~60 FPS GUI refresh

    def _render(self):
        ok, frame = self.cap.read()
        if not ok:
            self._set_status("Camera read failed")
            return

        # keep most recent frame accessible for sender thread
        self.current_frame = frame.copy()
        H, W = frame.shape[:2]
        self.last_frame_size = (W, H)

        # draw last known result depending on endpoint
        if self.endpoint_var.get() == "point":
            annotated = self._draw_points(frame, self.last_points, self.label_var.get())
        else:
            annotated = frame.copy()
            if self.last_caption:
                # wrap caption to ~half the width
                max_chars = max(10, int((W / 2) / 12))  # ~12 px/char heuristic
                wrapped = textwrap.wrap(self.last_caption, width=max_chars)

                y0 = 28
                for i, line in enumerate(wrapped):
                    y = y0 + i * 32  # line spacing
                    if y > H - 10:
                        break  # avoid drawing off-screen

                    # measure text and draw white background box
                    (text_w, text_h), _ = cv2.getTextSize(
                        line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    x1, y1 = 8, y - text_h - 6
                    x2, y2 = x1 + text_w + 8, y + 6
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W - 1, x2), min(H - 1, y2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), -1)

                    # draw text on top (red)
                    cv2.putText(
                        annotated,
                        line,
                        (x1 + 4, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

        # info text bottom-left with white background
        note = "Detection may take a few seconds to update"
        (tw, th), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x1, y1 = 8, H - th - 12
        x2, y2 = x1 + tw + 8, H - 4
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.putText(
            annotated,
            note,
            (x1 + 4, H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        # show
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        # if no request is in flight, try to send the latest frame
        if not self.in_flight:
            self._try_send_frame()

    def _draw_points(
        self, frame: np.ndarray, points: Optional[Iterable[Dict]], label: str
    ) -> np.ndarray:
        if not points:
            return frame
        out = frame.copy()
        H, W = out.shape[:2]

        for p in points:
            try:
                x, y = _to_xy(p, (W, H))
                # red "x" marker
                size = max(4, int(min(W, H) * 0.01))
                cv2.line(
                    out,
                    (int(x - size), int(y - size)),
                    (int(x + size), int(y + size)),
                    (0, 0, 255),
                    2,
                )
                cv2.line(
                    out,
                    (int(x - size), int(y + size)),
                    (int(x + size), int(y - size)),
                    (0, 0, 255),
                    2,
                )

                # label text with white background box
                tx, ty = int(x + 12), int(y - 12)  # text anchor (baseline-left)
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                pad = 4
                x1, y1 = tx - pad, ty - text_h - pad
                x2, y2 = tx + text_w + pad, ty + pad

                # clamp box into frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), -1)
                cv2.putText(
                    out,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            except Exception:
                continue
        return out

    def _try_send_frame(self, initial: bool = False):
        if not hasattr(self, "current_frame"):
            return
        if self.in_flight:
            return

        # capture the latest frame snapshot at send-time
        frame = self.current_frame.copy()
        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_bytes = jpg.tobytes()
        api = self.api_var.get()

        self.in_flight = True
        self._set_status("Sending…" if not initial else "Bootstrapping…")

        ep = self.endpoint_var.get()
        label = self.label_var.get()
        length = self.length_var.get()

        def worker():
            try:
                if ep == "point":
                    pts = _post_point(api, img_bytes, label)
                    self.last_points = pts
                    self.last_caption = None
                    self._set_status(f"OK ({len(pts)} point(s))")
                else:
                    cap_text = _post_caption(api, img_bytes, length)
                    self.last_caption = cap_text
                    self.last_points = None
                    self._set_status("OK (caption)")
            except Exception as e:
                self._set_status(f"Error: {e}")
            finally:
                self.in_flight = False  # allow next send

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraPointApp(root, cam_index=0)
    root.mainloop()
