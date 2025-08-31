import io
import logging
from enum import Enum
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from models import Moondream_FastVLM

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vlm-api")

# Initialize app
app = FastAPI(title="VLM API")
client = Moondream_FastVLM()


class CaptionLength(str, Enum):
    short = "short"
    normal = "normal"


def file_to_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


@app.get("/health")
def health():
    """Get application health status."""
    logger.info("Health check called")
    return {"status": "ok"}


@app.post("/caption")
def caption(
    file: UploadFile = File(...),
    length: CaptionLength = Form(default=CaptionLength.short),
):
    """Create image caption."""
    logger.info(f"/caption called with file={file.filename}, length={length}")
    img = file_to_image(file)
    text = client.caption(img, length=length)
    logger.debug(f"Caption result: {text}")
    return {"caption": text}


@app.post("/caption/stream")
def caption_stream(
    file: UploadFile = File(...),
    length: CaptionLength = Form(default=CaptionLength.short),
):
    """Stream image caption tokens."""
    logger.info(f"/caption/stream called with file={file.filename}, length={length}")
    img = file_to_image(file)

    def gen_stream():
        for tok in client.stream_caption(img, length=length):
            logger.debug(f"Stream caption token: {tok}")
            yield tok

    return StreamingResponse(gen_stream(), media_type="text/plain")


@app.post("/query")
def query(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Answer a question about an image."""
    logger.info(f"/query called with file={file.filename}, question={question!r}")
    img = file_to_image(file)
    ans = client.query(img, question)
    logger.debug(f"Query answer: {ans}")
    return {"answer": ans}


@app.post("/query/stream")
def query_stream(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Stream VQA answer tokens."""
    logger.info(
        f"/query/stream called with file={file.filename}, question={question!r}"
    )
    img = file_to_image(file)

    def gen_stream():
        for tok in client.stream_query(img, question):
            logger.debug(f"Stream query token: {tok}")
            yield tok

    return StreamingResponse(gen_stream(), media_type="text/plain")


@app.post("/detect")
def detect(
    file: UploadFile = File(...),
    label: str = Form(...),
):
    """Detect objects of a given label in an image and returns bounding boxes."""
    logger.info(f"/detect called with file={file.filename}, label={label}")
    img = file_to_image(file)
    objs = client.detect(img, label)
    logger.info(
        f"Detected {len(objs)} objects with file={file.filename}, label={label}"
    )
    logger.debug(f"Detected objects: {objs}")
    return {"objects": objs}


@app.post("/point")
def point(
    file: UploadFile = File(...),
    label: str = Form(...),
):
    """Locate points for a given label in an image."""
    logger.info(f"/point called with file={file.filename}, label={label}")
    img = file_to_image(file)
    pts = client.point(img, label)
    logger.debug(f"Located points: {pts}")
    return {"points": pts}
