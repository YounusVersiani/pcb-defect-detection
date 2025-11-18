# PCB Defect Detection

A production-ready system for automated detection and classification of defects in printed circuit board (PCB) images using deep learning and scalable APIs. This repository contains trained YOLOv8 model weights, an end-to-end inference pipeline, comprehensive analysis notebooks, and containerized deployment for local and cloud demonstration.

---

## Overview

- **Objective:** Accurate, automated detection and localization of PCB manufacturing defects.
- **Model:** YOLOv8 Nano, trained on a custom multi-class defect dataset.
- **API:** FastAPI-based REST endpoint supporting image upload and robust error handling.
- **Deployment:** Docker container available; public cloud deployment for demonstration (Render).
- **Notebooks:** Includes EDA and training analysis workflows.

---

## Model Details and Evaluation Metrics

- **Architecture:** YOLOv8 Nano (Ultralytics)
- **Training Data:** ~600 annotated PCB images (train, validation, test splits combined)
- **Classes:** 7 distinct PCB defect types
- **Performance:**
    - mAP@0.50: 99.49%
    - mAP@0.50-0.95: 83.08%
    - Precision: 99.33%
    - Recall: 99.99%
- **Model Size:** 6.2 MB

---

## API & Cloud Demo

A robust FastAPI service wraps the detector with a clear REST API.

- **Live API:**  
  [https://pcb-defect-detection-6moo.onrender.com](https://pcb-defect-detection-6moo.onrender.com)  
  - Interactive docs available at: [https://pcb-defect-detection-6moo.onrender.com/docs](https://pcb-defect-detection-6moo.onrender.com/docs)

### Core Endpoints

- `POST /predict`  
  Upload an image and receive detected defect locations and labels as a JSON response.
- `GET /health`  
  Service health and model status.
- `GET /docs`  
  Interactive Swagger (OpenAPI) documentation.

### How to Test the API Using Swagger UI

1. **Navigate to the interactive documentation:**  
   [https://pcb-defect-detection-6moo.onrender.com/docs](https://pcb-defect-detection-6moo.onrender.com/docs)

2. **Locate the `POST /predict` endpoint** and click to expand it.

3. **Click "Try it out"** button in the top-right corner of the endpoint section.

4. **Upload your PCB image:**
   - Click "Choose File" next to the `file` parameter
   - Select a PCB image from your local machine
   - The filename will appear once selected

5. **Click "Execute"** to run inference.
   - Note: Inference typically takes 20-30 seconds on the free-tier Render deployment

6. **View Results:**
   - Scroll down to the "Response body" section
   - JSON response includes:
     - `num_detections`: Total number of defects found
     - `detections`: Array of detected defects with:
       - `class_name`: Defect type classification
       - `confidence`: Detection confidence score
       - `bounding_box`: Spatial coordinates of the defect

#### Example Response Format

```json
{
  "filename": "pcb_sample.jpg",
  "image_size": {"width": 640, "height": 480},
  "num_detections": 2,
  "confidence_threshold": 0.5,
  "detections": [
    {
      "class_id": 0,
      "class_name": "defect_1",
      "confidence": 0.94,
      "bounding_box": {"x1": 120, "y1": 85, "x2": 195, "y2": 140}
    },
    {
      "class_id": 3,
      "class_name": "defect_4",
      "confidence": 0.89,
      "bounding_box": {"x1": 310, "y1": 220, "x2": 380, "y2": 275}
    }
  ]
}
```

#### Example: Inference via Python

```python
import requests
url = "https://pcb-defect-detection-6moo.onrender.com/predict"
files = {"file": open("pcb_image.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## Local Setup & Usage

1. **Prerequisites:** Python 3.11 or later.
2. **Install dependencies:**  
    ```
    pip install -r requirements.txt
    ```
3. **Run API server locally:**  
    ```
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    ```
    Access the API at [http://localhost:8000/docs](http://localhost:8000/docs)

### Docker Deployment

1. **Build and run with Docker:**  
    ```
    docker build -f docker/Dockerfile -t pcb-defect-api .
    docker run -p 8000:8000 pcb-defect-api
    ```

---

## Notebooks

The repository includes comprehensive Jupyter notebooks for analysis and training:

- **`01_eda.ipynb`**: Exploratory data analysis and dataset statistics
- **`02_training_analysis.ipynb`**: Training metrics, loss curves, and model evaluation

Located in the `notebooks/` directory.

---

## Customization and Retraining

- End-to-end notebooks provided under `notebooks/` for EDA, data preparation, and training analysis.
- To support additional classes or datasets, update the training notebooks and data configuration.
- Model weights and configuration are easily replaced for further fine-tuning or evaluation.

---

## License and Acknowledgments

MIT License.  
Key technologies: YOLOv8 (Ultralytics), FastAPI, Docker, Render.

