import torch
import timm

# ✅ Rebuild model
model = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=13)

# ✅ Load state_dict correctly
state_dict = torch.load("models/effbest.pt")
model.load_state_dict(state_dict)
model.eval()

# ✅ Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "effbest.onnx", opset_version=12)