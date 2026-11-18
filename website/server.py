"""Test page for the Leyenda Project workflow (classification, pre-processing, captioning), served on http://localhost:8000.

    python server.py [port]        (or start.bat on Windows, bash start.sh on Linux / macOS)

Only the Python standard library is used besides the models' own TensorFlow, NumPy and Pillow.
"""
import base64, io, json, sys, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
STATIC = {"/": ("index.html", "text/html"), "/style.css": ("style.css", "text/css"), "/app.js": ("app.js", "text/javascript")}
MAX_BYTES = 40 * 1024 * 1024
MODELS = threading.Lock()                             # one image at a time through the models; pages and files are served in parallel

print("loading the classifier, the denoiser and both captioners ...", flush=True)
import pipeline
if not pipeline.tf.config.list_physical_devices("GPU"):
    print("no GPU found: the chain runs on the CPU (about 20 s per photograph)", flush=True)
pipeline.warm_up()


def data_uri(img, side=480, quality=88):
    im = Image.fromarray(np.round(img * 255).astype(np.uint8)); im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fine_grain(img):
    # mean absolute Laplacian residual (0-255 scale): noise and fine texture
    g = img.mean(-1)
    return float(np.abs(g[1:-1, 1:-1] - 0.25 * (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])).mean() * 255)


def denoising_effect(img, cleaned):
    """What the denoiser changed, measured on the full-resolution images."""
    before, after = fine_grain(img), fine_grain(cleaned)
    return {"mean_change": float(np.abs(cleaned - img).mean() * 255), "grain_before": before, "grain_after": after}


class Handler(BaseHTTPRequestHandler):
    timeout = 60                                      # browsers open spare connections that may never send a request

    def send(self, code, body, kind):
        self.send_response(code); self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path not in STATIC:
            return self.send(404, b"not found", "text/plain")
        name, kind = STATIC[self.path]
        self.send(200, (HERE / name).read_bytes(), kind)

    def do_POST(self):
        if self.path != "/process":
            return self.send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        name = unquote(self.headers.get("X-Filename", "image"))
        if n > MAX_BYTES:
            self.rfile.read(n)
            return self.reply({"name": name, "error": f"file larger than {MAX_BYTES // 2**20} MB"})
        data = self.rfile.read(n)
        t = time.time()
        try:
            first = {"123": False, "213": True}.get(self.headers.get("X-Order"), pipeline.DENOISE_FIRST)
            with MODELS:
                r = pipeline.run(data, first)
        except Exception as e:                        # not an image, or a format Pillow cannot read
            print("error on", name, repr(e), flush=True)
            return self.reply({"name": name, "error": "not a readable image file"})
        h, w = r["input"].shape[:2]
        cleaned = r["denoised"]
        self.reply({"name": name, "size": f"{w} x {h}", "input": data_uri(r["input"]),
                    "denoised": data_uri(cleaned) if cleaned is not None else None,
                    "input_full": data_uri(r["input"], 1024, 95),          # for the side-by-side view at full size
                    "denoised_full": data_uri(cleaned, 1024, 95) if cleaned is not None else None,
                    "effect": denoising_effect(r["input"], cleaned) if cleaned is not None else None,
                    "p_photo": r["p_photo"], "photo": bool(r["photo"]), "captions": r["captions"],
                    "denoise_first": first, "seconds": round(time.time() - t, 2)})

    def reply(self, obj):
        self.send(200, json.dumps(obj).encode(), "application/json")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"ready: http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
