# ...existing code...
import os
import torch
import timm

# Config
MODEL_NAME = "mobilenetv3_small_100"  # or "mobilenetv3_large_100"
NUM_CLASSES = 13
IMG_SIZE = (1, 3, 224, 224)
CHECKPOINT_IN = "models/mobilenetv3_ckpt.pt"   # optional input checkpoint
STATE_OUT = "models/mobilenetv3_state.pt"      # saved state_dict
SCRIPTED_OUT = "models/mobilenetv3_scripted.pt"  # scripted model for mobile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
model.to(device)
model.eval()

# Load checkpoint if present (supports raw state_dict or dict with 'state_dict' key)
if os.path.exists(CHECKPOINT_IN):
    ckpt = torch.load(CHECKPOINT_IN, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    # Allow partial loading if layer names differ (set strict=False if needed)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

# Save state_dict (recommended)
os.makedirs(os.path.dirname(STATE_OUT), exist_ok=True)
torch.save(model.state_dict(), STATE_OUT)

# Optionally create a scripted model for mobile deployment
# Use tracing (fast) or scripting (more general). Here we try scripting first.
try:
    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.script(model_cpu)
except Exception:
    # fallback to tracing if scripting fails
    dummy = torch.randn(*IMG_SIZE)
    scripted = torch.jit.trace(model_cpu, dummy)

# Try lightweight mobile optimization if available
try:
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized = optimize_for_mobile(scripted._c)
    torch.jit.save(optimized, SCRIPTED_OUT)
except Exception:
    # save scripted without extra mobile optimization
    torch.jit.save(scripted, SCRIPTED_OUT)

print(f"Saved state_dict -> {STATE_OUT}")
print(f"Saved scripted model -> {SCRIPTED_OUT}")
# ...existing code...