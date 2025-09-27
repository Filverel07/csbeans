from ultralytics import YOLO

# 🧩 Paths and settings
model = YOLO('runs/classify/yolov12_cls_scratch/weights/last.pt')
data_path = 'datasets/defect_split_cls'
training_name = 'yolov12_cls_scratch'
num_epochs = 250


# 🚀 Start training
results = model.train(
    data=data_path,
    epochs=num_epochs,
    imgsz=224,
    batch=16,
    name=training_name,
    patience=15,
    exist_ok=True,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.01,
    mosaic=False,
    mixup=False,
    auto_augment='randaugment',
    amp=True,
    pretrained=True,
    dropout=0.2,
    erasing=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    seed=0,
    deterministic=True,
    plots=True,
    val=True,
    split='val',
    save=True,
    save_period=-1,
    workers=8,
    cos_lr=True,
    resume=True, #resume training from last.pt
)