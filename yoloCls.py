from ultralytics import YOLO

# 🧩 Paths and settings
model = YOLO('runs/classify/yolov12_cls_detNew/weights/best.pt')
data_path = 'datasets/224_v3_2__balanced_split'
training_name = '224_v3_2_fineTuningv3'
num_epochs = 250


results = model.train(
    data=data_path,
    epochs=num_epochs,
    imgsz=224,
    batch=64,
    name=training_name,
    patience=15,
    exist_ok=True,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    mosaic=False,
    mixup=False,
    auto_augment=None,     # ❌ disable RandAugment
    amp=False,
    pretrained=False,
    dropout=0.0,           # ❌ disable dropout
    erasing=0.0,           # ❌ disable random erasing
    fliplr=0.0,            # ❌ disable horizontal flip
    translate=0.0,         # ❌ disable translation
    scale=0.0,             # ❌ disable scaling
    seed=0,
    deterministic=True,
    plots=True,
    val=True,
    split='val',
    save=True,
    save_period=-1,
    workers=8,
    cos_lr=True,
    #resume=True,            # ✅ resume training if interrupted
)