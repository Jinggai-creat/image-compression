import math
import torch
import torch.nn as nn

from gdn import GDN
from bit_estimator import BitEstimator
import utils


class AnalysisNet(nn.Module):
    """AnalysisNet"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(AnalysisNet, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.gdn1 = GDN(out_channels_n)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.gdn2 = GDN(out_channels_n)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.gdn3 = GDN(out_channels_n)

        self.conv4 = nn.Conv2d(out_channels_n, out_channels_m, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x


class AnalysisPriorNet(nn.Module):
    """AnalysisPriorNet"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(AnalysisPriorNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channels_m, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class SynthesisNet(nn.Module):
    """SynthesisNet"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(SynthesisNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_m, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(out_channels_n, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(out_channels_n, inverse=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn3 = GDN(out_channels_n, inverse=True)

        self.deconv4 = nn.ConvTranspose2d(out_channels_n, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x


class SynthesisPriorNet(nn.Module):
    """SynthesisPriorNet"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(SynthesisPriorNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_m, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        x = torch.exp(x)
        return x


class EDICImageCompression(nn.Module):
    """EDICImageCompression"""
    def __init__(self, out_channels_n=192, out_channels_m=320):
        super(EDICImageCompression, self).__init__()
        self.out_channels_n = out_channels_n
        self.out_channels_m = out_channels_m

        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)

        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorNet(out_channels_n, out_channels_m)

        self.bit_estimator_z = BitEstimator(out_channels_n)

    def iclr18_estimate_bits_z(self, z):
        prob = self.bit_estimator_z(z + 0.5) - self.bit_estimator_z(z - 0.5)
        total_bits = torch.sum(
            torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50)
        )
        return total_bits, prob

    def forward(self, x):
        # step 1: init
        quant_noise_feature = torch.zeros(
            x.size(0),
            self.out_channels_m, x.size(2) // 16, x.size(3) // 16
        ).to(x.device)
        quant_noise_z = torch.zeros(
            x.size(0), 
            self.out_channels_n, x.size(2) // 64, x.size(3) // 64
        ).to(x.device)

        quant_noise_feature = torch.nn.init.uniform_(
            torch.zeros_like(quant_noise_feature), 
            -0.5, 0.5
        )
        quant_noise_z = torch.nn.init.uniform_(
            torch.zeros_like(quant_noise_z), 
            -0.5, 0.5
        )

        # step 2: input image into encoder
        feature = self.encoder(x)
        batch_size = feature.size()[0]

        # step 3: input feature into prior(entropy encoding)
        z = self.encoder_prior(feature)
        # TODO(hujiakui): may need 3 + 4, like my deblurgan
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        # step 4: get the feature output the prior(entropy decoding)
        recon_sigma = self.decoder_prior(compressed_z)
        feature_renorm = feature

        # step 5: get the image from decoder
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.decoder(compressed_feature_renorm)
        clipped_recon_image = recon_image.clamp(0., 1.)       # because it's float, clamp it

        total_bits_feature, _ = utils.feature_probs_based_sigma(
            compressed_feature_renorm, recon_sigma
        )
        total_bits_z, _ = self.iclr18_estimate_bits_z(compressed_z)

        x_shape = x.size()
        bpp_feature = total_bits_feature / (batch_size * x_shape[2] * x_shape[3])
        bpp_z = total_bits_z / (batch_size * x_shape[2] * x_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, bpp
