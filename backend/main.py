from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil

from backend.utils.inference import run_inference

# -------------------------------------------------
# App init
# -------------------------------------------------
app = FastAPI(
    title="Skin Cancer Detection System",
    description="YOLOv8-based skin lesion analysis",
    version="1.0.0"
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UI_FILE = BASE_DIR / "ui.html"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Static files (images)
# -------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Serve UI (single-file dark UI)
# -------------------------------------------------
@app.get("/")
def serve_ui():
    """
    Serves the single-page developer UI (ui.html)
    """
    return FileResponse(UI_FILE)

# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, runs YOLO inference,
    returns output image + predictions
    """

    # Save uploaded image
    image_path = UPLOAD_DIR / file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    output_image, detections = run_inference(image_path)

    # Response
    return JSONResponse({
        "original": f"/static/uploads/{file.filename}",
        "result": f"/static/outputs/{output_image}",
        "detections": detections
    })
