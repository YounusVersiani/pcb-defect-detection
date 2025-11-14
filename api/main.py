"""
FastAPI Application for PCB Defect Detection
Upload PCB image and receive defect detection results
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="PCB Defect Detection API",
    description="Upload PCB images to detect manufacturing defects using YOLOv8",
    version="1.0.0"
)

MODEL_PATH = "runs/detect/pcb_defect_yolov8n6/weights/best.pt"
model = None


@app.on_event("startup")
async def load_model():
    """Load YOLOv8 model on API startup"""
    global model
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "PCB Defect Detection API",
        "status": "running",
        "endpoints": {
            "POST /predict": "Upload image for defect detection",
            "GET /health": "Check API health status",
            "GET /docs": "Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH
    }


@app.post("/predict")
async def predict_defects(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """
    Detect defects in uploaded PCB image
    
    Args:
        file: Image file (JPG, PNG)
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
    
    Returns:
        JSON with detection results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.0 and 1.0"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        results = model.predict(
            source=image_array,
            conf=confidence_threshold,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        detections = []
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().tolist()
            
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": round(bbox[0], 2),
                    "y1": round(bbox[1], 2),
                    "x2": round(bbox[2], 2),
                    "y2": round(bbox[3], 2)
                }
            })
        
        return {
            "filename": file.filename,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "num_detections": len(detections),
            "confidence_threshold": confidence_threshold,
            "detections": detections
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)