import torch
print(torch.version.cuda)         # Should be 11.7 or newer
print(torch.cuda.get_device_name(0))
