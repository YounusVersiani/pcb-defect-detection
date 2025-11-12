"""
YOLOv8 Training Script for PCB Defect Detection
Trains a YOLOv8 nano model on PCB defect dataset
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import yaml


def train_yolov8_pcb_defect():
    """Train YOLOv8 nano model on PCB defect dataset"""
    
    print("\n" + "="*70)
    print("YOLOv8 PCB DEFECT DETECTION TRAINING")
    print("="*70 + "\n")
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.2f} GB")
    
    print()
    
    # Load pretrained model
    print("Loading YOLOv8 nano pretrained model...")
    model = YOLO('yolov8n.pt')
    print("Model loaded successfully\n")
    
    # Verify dataset config
    data_yaml = Path('data/raw/data.yaml')
    
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml.resolve()}")
        print("Make sure dataset is downloaded to data/raw/")
        return
    
    print(f"Dataset config: {data_yaml.resolve()}")
    
    # Read dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Classes: {len(data_config['names'])}")
    for i, class_name in enumerate(data_config['names']):
        print(f"  Class {i}: {class_name}")
    
    print()
    
    # Training configuration
    print("Training Configuration:")
    print("  Model: YOLOv8n (nano)")
    print("  Epochs: 50")
    print("  Image Size: 640x640")
    print("  Batch Size: 4")
    print("  Patience: 10 (early stopping)")
    print()
    
    print("Starting training...\n")
    print("-"*70)
    
    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=4,
        device=device,
        name='pcb_defect_yolov8n',
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        seed=42,
        mosaic=0.0,
        mixup=0.0
    )
    
    print("-"*70)
    
    # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    results_dir = Path('runs/detect/pcb_defect_yolov8n')
    weights_dir = results_dir / 'weights'
    
    print(f"Results saved to: {results_dir}\n")
    print("Output files:")
    print(f"  Best model: {weights_dir / 'best.pt'}")
    print(f"  Last checkpoint: {weights_dir / 'last.pt'}")
    print(f"  Training plots: {results_dir / 'results.png'}")
    print(f"  Confusion matrix: {results_dir / 'confusion_matrix.png'}\n")
    
    # Final metrics
    print("Final Training Metrics:")
    print("-"*70)
    
    metrics_dict = results.results_dict
    
    if 'metrics/mAP50(B)' in metrics_dict:
        print(f"  mAP@50: {metrics_dict['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in metrics_dict:
        print(f"  mAP@50-95: {metrics_dict['metrics/mAP50-95(B)']:.4f}")
    if 'metrics/precision(B)' in metrics_dict:
        print(f"  Precision: {metrics_dict['metrics/precision(B)']:.4f}")
    if 'metrics/recall(B)' in metrics_dict:
        print(f"  Recall: {metrics_dict['metrics/recall(B)']:.4f}")
    
    print("-"*70)
    print("\nModel training completed successfully.\n")
    
    return results


if __name__ == "__main__":
    try:
        results = train_yolov8_pcb_defect()
    except Exception as e:
        print(f"\nERROR during training: {e}")
        print("Please check dataset path and GPU configuration.")

