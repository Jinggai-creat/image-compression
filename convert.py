from typing import Tuple
import torch
from torch.nn.functional import batch_norm
import torch.onnx

from models import *


class EDICConvert(EDICImageCompression):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super().__init__(out_channels_n=out_channels_n, out_channels_m=out_channels_m)
    
    def forward(self, x):
        feature = self.encoder(x)
        compressed_feature_renorm = torch.round(feature)
        recon_image = self.decoder(compressed_feature_renorm)
        return recon_image


# models init
model = EDICConvert().cuda()
model_params = torch.load("pretrain.pth")
model.load_state_dict(model_params)
model.eval()

# modify the shape of input
dummy_input = torch.randn(1, 1, 64, 64).cuda()
input_names = ["image_input"]
output_names = ["image_output"]

torch.onnx.export(
    model, dummy_input, "edic.onnx", 
    verbose=True, 
    input_names=input_names, output_names=output_names, 
    opset_version=11
)
