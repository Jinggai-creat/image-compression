import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gdn import GDN
from bit_estimator import BitEstimator
import utils


class MaxBlurPool(nn.Module):
    """
    Simplified implementation of MaxBlurPool
    """
    def __init__(self, n):
        super(MaxBlurPool, self).__init__()
        self.maxpool = nn.MaxPool2d((2, 2))
        self.padding = nn.ReflectionPad2d(1)

        f = torch.tensor([1, 2, 1])
        f = (f[None, :] * f[:, None]).float()
        f /= f.sum()
        f = f[None, None].repeat((n, 1, 1, 1))

        self.register_buffer('f', f)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.padding(x)
        x = F.conv2d(x, self.f, stride=2, groups=x.shape[1])
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_features: int = 128, reduction: int = 8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1, bias=True),
            nn.CELU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        return score, x * score
        

class AnalysisNet(nn.Module):
    """AnalysisNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(AnalysisNet, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels_n, kernel_size=5, stride=2, padding=2)
        nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channels_n) / (6))))
        nn.init.constant_(self.conv1.bias.data, 0.01)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.gdn1 = GDN(out_channels_n)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        nn.init.constant_(self.conv2.bias.data, 0.01)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.gdn2 = GDN(out_channels_n)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channels_n)

        self.conv4 = nn.Conv2d(out_channels_n, out_channels_m, kernel_size=5, stride=2, padding=2)
        nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channels_m + out_channels_n) / (out_channels_n + out_channels_n))))
        nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        res = self.pool1(x)
        x = self.gdn2(self.conv2(x)) + res
        res = self.pool2(x)
        x = self.gdn3(self.conv3(x)) + res
        x = self.conv4(x)
        return x


class AnalysisPriorNet(nn.Module):
    """AnalysisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(AnalysisPriorNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channels_m, out_channels_n, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channels_m + out_channels_n) / (out_channels_m + out_channels_m))))
        nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        nn.init.constant_(self.conv2.bias.data, 0.01)

    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class SynthesisNet(nn.Module):
    """SynthesisNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(SynthesisNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_m, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * (out_channels_m + out_channels_n) / (out_channels_m + out_channels_m))))
        nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channels_n, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2))
        nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channels_n, inverse=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2))
        nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn3 = GDN(out_channels_n, inverse=True)

        self.deconv4 = nn.ConvTranspose2d(out_channels_n, 1, kernel_size=5, stride=2, padding=2, output_padding=1)
        nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * (out_channels_n + 3) / (out_channels_n + out_channels_n))))
        nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        res = F.interpolate(x, scale_factor=2.)
        x = self.igdn2(self.deconv2(x)) + res
        res = F.interpolate(x, scale_factor=2.)
        x = self.igdn3(self.deconv3(x)) + res
        x = self.deconv4(x)
        return x


class SynthesisPriorNet(nn.Module):
    """SynthesisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(SynthesisPriorNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2))
        nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2))
        nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_m, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * (out_channels_m + out_channels_n) / (out_channels_n + out_channels_n))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        x = torch.exp(x)
        return x


class SynthesisPriorCANet(SynthesisPriorNet):
    """SynthesisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(SynthesisPriorCANet, self).__init__()
        self.ca = ChannelAttention(num_features=out_channels_m)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        mu, x = self.ca(x)
        return mu, x


class EDICImageCompression(nn.Module):
    """EDICImageCompression"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICImageCompression, self).__init__()
        self.out_channels_n = out_channels_n
        self.out_channels_m = out_channels_m

        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)

        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)
        # self.decoder_prior = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)

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
        ).to(x.device)
        quant_noise_z = torch.nn.init.uniform_(
            torch.zeros_like(quant_noise_z), 
            -0.5, 0.5
        ).to(x.device)

        # step 2: input image into encoder
        feature = self.encoder(x)
        batch_size = feature.size()[0]

        # step 3: put feature into prior(entropy encoding)
        z = self.encoder_prior(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        # step 4: compress the feature by the prior(entropy decoding)
        # recon_sigma = self.decoder_prior(compressed_z)
        recon_mu, recon_sigma = self.decoder_prior(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        # step 5: get the image from decoder
        recon_image = self.decoder(compressed_feature_renorm)
        # clipped_recon_image = recon_image.clamp(0., 1.)       # because it's float, clamp it

        total_bits_feature, _ = utils.feature_probs_based_sigma(
            compressed_feature_renorm, recon_sigma, recon_mu
        )
        total_bits_z, _ = self.iclr18_estimate_bits_z(compressed_z)

        x_shape = x.size()
        bpp_feature = total_bits_feature / (batch_size * x_shape[2] * x_shape[3])
        bpp_z = total_bits_z / (batch_size * x_shape[2] * x_shape[3])

        return recon_image, bpp_feature, bpp_z


if __name__ == "__main__":
    a = EDICImageCompression().cuda()
    test_data = torch.rand(1, 1, 64, 64).cuda()
    a(test_data)
