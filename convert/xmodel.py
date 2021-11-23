"""
use Vitis-ai: 1.4.0
"""
import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import sys
sys.path.append("..")

from models import *


class EDICEncoder(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICEncoder, self).__init__()
        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)

    def forward(self, x):
        feature = self.encoder(x)
        feature = torch.round(feature)
        z = self.encoder_prior(feature)
        z = torch.round(z)
        return feature, z


class EDICDecoder(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICDecoder, self).__init__()
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)
    
    def forward(self, feature, z):
        mu = self.decoder_prior_mu(z)
        feature = feature - mu
        delta = torch.floor(feature + 0.5) - feature
        feature = feature + delta
        feature = feature + mu
        recon_image = self.decoder(feature)
        return recon_image


class EDICConvert(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICConvert, self).__init__()
        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)

    def forward(self, x):
        feature = self.encoder(x)
        feature = torch.round(feature)
        z = self.encoder_prior(feature)
        z = torch.round(z)
        mu = self.decoder_prior_mu(z)
        feature = feature - mu
        delta = torch.floor(feature + 0.5) - feature
        feature = feature + delta
        feature = feature + mu
        recon_image = self.decoder(feature)
        return recon_image


def model_info_read(model):
    for name, buf in model.named_buffers():
        if 'anchor_grid' in name:
            register_buffers = {'anchor_grid': buf}
    return register_buffers


def quantize(quant_mode: str = "calib", device=torch.device("cpu")):
    edic_model = torch.load("../model/pretrain_converted_low.pth")

    # need ==> edic_encoder + edic_decoder
    # edic_encoder = None
    # edic_decoder = None

    # ================================ Quantizer API ====================================
    # ===================================================================================
    img = torch.randn([1, 1, 128, 128]) * 255
    quantizer = torch_quantizer(
        quant_mode, edic_model, (img), device=device
    )
    quant_model = quantizer.quant_model
    recon_img = quant_model(img)
    print("MSE Loss: {}".format(nn.MSELoss()(img, recon_img)))

    quantizer.export_quant_config()

    # feature = torch.randn([1, 192, 32, 32])
    # z = torch.randn([1, 128, 8, 8])
    # encoder_quantizer = torch_quantizer(
    #     "calib", edic_encoder, (img), device=device
    # )
    # quant_encoder = encoder_quantizer.quant_model

    # decoder_quantizer = torch_quantizer(
    #     "calib", edic_decoder, (feature, z), device=device
    # )
    # quant_decoder = encoder_quantizer.quant_model


if __name__ == "__main__":
    quantize()
