import torch
import torch.onnx

import models


# models init
model = models.EDICImageCompression()
model_params = torch.load("edic_epoch_3_bpp_0.080681.pth")
model.load_state_dict(model_params)
model.eval()

# modify the shape of input
dummy_input = torch.randn(1, 3, 256, 256)
input_names = ["image_input"]
output_names = ["image_output", "bpp_feature", "bpp_z"]

torch.onnx.export(
	model, dummy_input, "edic.onnx", 
	verbose=True, 
	input_names=input_names, output_names=output_names, 
	opset_version=11
)
