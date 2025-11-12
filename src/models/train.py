"""
YOLOv8 Training Script for PCB Defect Detection
Trains a YOLOv8 model to detect defects on circuit boards
"""

from ultralytics import YOLO
from pathlib import Path
import torch
import yaml

def train_yolov8_pcb_defect():
    """
    Train YOLOv8 nano model on PCB defect dataset
    """
    print("\n" + "=" * 70)
    print("🚀 YOLOv8 PCB DEFECT DETECTION TRAINING".center(70))
    print("=" * 70 + "\n")
    
    # ============================================================
    # 1. CHECK GPU AVAILABILITY
    # ============================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device.upper()}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Name: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
    
    print()
    
    # ============================================================
    # 2. LOAD PRETRAINED YOLOV8 MODEL
    # ============================================================
    print("📥 Loading YOLOv8 nano pretrained model...")
    model = YOLO('yolov8n.pt')  # Downloads automatically first time
    print("✅ Model loaded successfully\n")
    
    # ============================================================
    # 3. VERIFY DATASET CONFIG
    # ============================================================
    data_yaml = Path('data/raw/data.yaml')
    
    if not data_yaml.exists():
        print(f"❌ ERROR: data.yaml not found at {data_yaml.resolve()}")
        print("Make sure dataset is downloaded to data/raw/")
        return
    
    print(f"📊 Dataset Config: {data_yaml.resolve()}")
    
    # Read and display dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"   Classes: {len(data_config['names'])}")
    for i, class_name in enumerate(data_config['names']):
        print(f"     - Class {i}: {class_name}")
    
    print()
    
    # ============================================================
    # 4. TRAINING CONFIGURATION
    # ============================================================
    print("⚙️  Training Configuration:")
    print("   Model Architecture: YOLOv8n (nano)")
    print("   Epochs: 50")
    print("   Image Size: 640x640 pixels")
    print("   Batch Size: 16")
    print("   Learning Rate: default (0.01)")
    print("   Patience: 10 (early stopping)")
    print("   Device: " + device.upper())
    print()
    
    # ============================================================
    # 5. START TRAINING
    # ============================================================
    print("🚀 Starting training...\n")
    print("-" * 70)
    
    results = model.train(
        data=str(data_yaml),           # Path to dataset config
        epochs=50,                      # Number of training epochs
        imgsz=640,                      # Input image size (640x640)
        batch=16,                       # Batch size per GPU
        device=device,                  # Use GPU if available
        name='pcb_defect_yolov8n',     # Experiment name
        patience=10,                    # Early stopping patience
        save=True,                      # Save checkpoints and best model
        plots=True,                     # Generate training plots
        verbose=True,                   # Detailed logging
        workers=4,                      # Data loading workers
        seed=42                         # For reproducibility
    )
    
    print("-" * 70)
    
    # ============================================================
    # 6. TRAINING COMPLETE
    # ============================================================
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!".center(70))
    print("=" * 70 + "\n")
    
    results_dir = Path('runs/detect/pcb_defect_yolov8n')
    weights_dir = results_dir / 'weights'
    
    print("📂 Results Location:")
    print(f"   {results_dir}\n")
    
    print("📊 Output Files:")
    print(f"   ✅ Best Model: {weights_dir / 'best.pt'}")
    print(f"   ✅ Last Checkpoint: {weights_dir / 'last.pt'}")
    print(f"   ✅ Training Plots: {results_dir / 'results.png'}")
    print(f"   ✅ Confusion Matrix: {results_dir / 'confusion_matrix.png'}\n")
    
    # ============================================================
    # 7. PRINT FINAL METRICS
    # ============================================================
    print("📈 Final Training Metrics:")
    print("-" * 70)
    
    # Extract key metrics from results
    metrics_dict = results.results_dict
    
    if 'metrics/mAP50(B)' in metrics_dict:
        print(f"   mAP@50 (50% IoU):     {metrics_dict['metrics/mAP50(B)']:.4f}")
    if 'metrics/mAP50-95(B)' in metrics_dict:
        print(f"   mAP@50-95 (all IoU):  {metrics_dict['metrics/mAP50-95(B)']:.4f}")
    if 'metrics/precision(B)' in metrics_dict:
        print(f"   Precision:            {metrics_dict['metrics/precision(B)']:.4f}")
    if 'metrics/recall(B)' in metrics_dict:
        print(f"   Recall:               {metrics_dict['metrics/recall(B)']:.4f}")
    
    print("-" * 70)
    print("\n🎉 Your PCB defect detection model is ready to use!\n")
    
    return results


if __name__ == "__main__":
    try:
        results = train_yolov8_pcb_defect()
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        print("Please check your dataset path and GPU configuration.")
