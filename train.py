import argparse
from ultralytics import YOLO
import os
import sys
import torch

# Training parameters
EPOCHS = 500  # Increased epochs for better convergence
BATCH_SIZE = 50  # Increased batch size
IMGSZ = 640  # Image size
OPTIMIZER = 'AdamW'  # Optimizer choice
MOMENTUM = 0.937  # Momentum for optimizer
LR0 = 0.001  # Initial learning rate
LRF = 0.01  # Final learning rate factor
WEIGHT_DECAY = 0.0005  # Weight decay
WARMUP_EPOCHS = 5  # Warmup epochs
WARMUP_MOMENTUM = 0.8  # Warmup momentum
WARMUP_BIAS_LR = 0.1  # Warmup bias learning rate
DEVICE = '0'  # Default to GPU 0
WORKERS = 4  # Number of workers
AMP = True  # Automatic mixed precision

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate factor')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=WARMUP_EPOCHS, help='Warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=WARMUP_MOMENTUM, help='Warmup momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=WARMUP_BIAS_LR, help='Warmup bias learning rate')
    parser.add_argument('--device', type=str, default=DEVICE, help='Device to use (e.g., 0 for GPU 0)')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Number of workers')
    parser.add_argument('--amp', action='store_true', default=AMP, help='Use automatic mixed precision')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Initialize model
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Load YOLOv11n model
    model = YOLO('yolo11n.pt')
    
    # Training configuration
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        patience=100,  # Early stopping patience
        save=True,  # Save checkpoints
        save_period=10,  # Save every 10 epochs
        cache=False,  # No caching
        workers=args.workers,
        project='runs/train',
        name='yolo11n_advanced',  # Updated name to match model
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,  # For reproducibility
        deterministic=True,
        amp=args.amp,  # Automatic mixed precision
        val=True,  # Run validation
        plots=True,  # Generate plots
        save_dir='runs/train/yolo11n_advanced'  # Save directory
    )


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''