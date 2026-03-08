"""Create an initial sine checkpoint for shakeback."""

import torch
from problem import SineNet

model = SineNet(hidden=64)
torch.save({"model_state_dict": model.state_dict(), "hidden": 64},
           "sine_init.pt")
print("Saved sine_init.pt")
