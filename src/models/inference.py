"""
YOLOv8 Inference Script for PCB Defect Detection
Load trained model and run predictions on test images
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


def load_model(weights_path):
    """
    Load trained YOLOv8 model from weights file
    
    Args:
        weights_path: Path to .pt model file
        
    Returns:
        Loaded YOLO model
    """
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    
    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    print(f"Model loaded successfully")
    print(f"Model classes: {model.names}")
    
    return model


def run_inference(model, image_path, conf_threshold=0.25, save_dir=None):
    """
    Run inference on a single image
    
    Args:
        model: Loaded YOLO model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections (default 0.25)
        save_dir: Directory to save annotated images (optional)
        
    Returns:
        Detection results
    """
    print(f"\nRunning inference on: {image_path}")
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )
    
    result = results[0]
    boxes = result.boxes
    num_detections = len(boxes)
    
    print(f"Detections found: {num_detections}")
    
    if num_detections > 0:
        print("\nDetection Details:")
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            print(f"  [{i+1}] {class_name} - Confidence: {confidence:.3f} - BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
    if save_dir:
        save_path = Path(save_dir) / f"predicted_{Path(image_path).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        annotated_img = result.plot()
        cv2.imwrite(str(save_path), annotated_img)
        print(f"\nAnnotated image saved to: {save_path}")
    
    return result


def run_batch_inference(model, image_dir, conf_threshold=0.25, save_dir=None):
    """
    Run inference on multiple images in a directory
    
    Args:
        model: Loaded YOLO model
        image_dir: Directory containing input images
        conf_threshold: Confidence threshold for detections
        save_dir: Directory to save annotated images (optional)
        
    Returns:
        List of detection results
    """
    image_dir = Path(image_dir)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f'*{ext}'))
        image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return []
    
    print(f"\nFound {len(image_paths)} images in {image_dir}")
    print("="*70)
    
    results = []
    for img_path in image_paths:
        result = run_inference(model, str(img_path), conf_threshold, save_dir)
        results.append(result)
        print("-"*70)
    
    return results


def main():
    """Main inference function"""
    
    print("\n" + "="*70)
    print("YOLOv8 PCB DEFECT DETECTION - INFERENCE")
    print("="*70)
    
    WEIGHTS_PATH = "runs/detect/pcb_defect_yolov8n6/weights/best.pt"
    TEST_IMAGE_DIR = "data/raw/test/images"
    SAVE_DIR = "runs/inference/predictions"
    CONFIDENCE_THRESHOLD = 0.5
    
    model = load_model(WEIGHTS_PATH)
    
    print(f"\nRunning inference on test images...")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Saving results to: {SAVE_DIR}")
    
    results = run_batch_inference(
        model=model,
        image_dir=TEST_IMAGE_DIR,
        conf_threshold=CONFIDENCE_THRESHOLD,
        save_dir=SAVE_DIR
    )
    
    print("\n" + "="*70)
    print("INFERENCE SUMMARY")
    print("="*70)
    
    total_images = len(results)
    images_with_detections = sum(1 for r in results if len(r.boxes) > 0)
    total_detections = sum(len(r.boxes) for r in results)
    
    print(f"\nTotal images processed: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total defects detected: {total_detections}")
    
    if total_images > 0:
        print(f"Average detections per image: {total_detections / total_images:.2f}")
    
    print(f"\nAnnotated images saved in: {SAVE_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()

