from ultralytics import settings
import lightly_train

lightly_train.train(
    out="out/beanDefect_Lightly",           # Output directory for trained model
    data="datasets/beans_split/train",      # Your dataset folder with images
    model="ultralytics/yolov12n.yaml",      # YOLOv12 model config
    epochs=30,                              # Number of epochs
    batch_size=32,                          # Batch size
    overwrite=True,
    num_workers=8
)


#pip install lightly lightly_train