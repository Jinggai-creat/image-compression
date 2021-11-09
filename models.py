import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gdn_converted import GDN
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
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        return score, x * score


class ResBlock(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 3, concat: bool = True):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        if concat:
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=1)
            self.norm1 = nn.InstanceNorm2d(channels, affine=False, track_running_stats=True)
            self.norm2 = nn.InstanceNorm2d(channels, affine=False, track_running_stats=True)
        else:
            self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size, stride=1)
            self.conv2 = nn.Conv2d(channels // 2, channels // 2, kernel_size, stride=1)
            self.norm1 = nn.InstanceNorm2d(channels // 2, affine=False, track_running_stats=True)
            self.norm2 = nn.InstanceNorm2d(channels // 2, affine=False, track_running_stats=True)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.norm1(x)
        res = x
        x = self.activation(x)

        x = self.pad(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return torch.cat([res, x], dim=1)


class ChannelNorm2d(nn.Module):
    def __init__(self, in_channels: int = 64, momentum=1e-1, affine=True, eps=1e-3):
        super().__init__()
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        """
        Calculate moments over channel dim, normalize.
        x:  Image tensor, shape (B,C,H,W)
        """
        mu, var = torch.mean(x, dim=1, keepdim=True), torch.var(
            x, dim=1, keepdim=True)

        x_normed = (x - mu) * torch.rsqrt(var + self.eps)

        if self.affine:
            x_normed = self.gamma * x_normed + self.beta
        return x_normed
        
# ---------
# EDIC
# ---------
class AnalysisNet(nn.Module):
    """AnalysisNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(AnalysisNet, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.gdn1 = GDN(out_channels_n)
        self.in1 = nn.InstanceNorm2d(out_channels_n)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.gdn2 = GDN(out_channels_n)
        self.in2 = nn.InstanceNorm2d(out_channels_n)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.gdn3 = GDN(out_channels_n)
        self.in3 = nn.InstanceNorm2d(out_channels_n)

        self.conv4 = nn.Conv2d(out_channels_n, out_channels_m, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.gdn1(self.in1(self.conv1(x)))
        res = self.pool1(x)
        x = self.gdn2(self.in2(self.conv2(x))) + res
        res = self.pool2(x)
        x = self.gdn3(self.in3(self.conv3(x))) + res
        x = self.conv4(x)
        return x


class AnalysisPriorNet(nn.Module):
    """AnalysisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(AnalysisPriorNet, self).__init__()
        self.conv1 = nn.Conv2d(out_channels_m, out_channels_n, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.conv3 = nn.Conv2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2)

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
        self.igdn1 = GDN(out_channels_n, inverse=True)
        self.in1 = nn.InstanceNorm2d(out_channels_n)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(out_channels_n, inverse=True)
        self.in2 = nn.InstanceNorm2d(out_channels_n)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.igdn3 = GDN(out_channels_n, inverse=True)
        self.in3 = nn.InstanceNorm2d(out_channels_n)

        self.deconv4 = nn.ConvTranspose2d(out_channels_n, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.igdn1(self.in1(self.deconv1(x)))
        res = F.interpolate(x, scale_factor=2.)
        x = self.igdn2(self.in2(self.deconv2(x))) + res
        res = F.interpolate(x, scale_factor=2.)
        x = self.igdn3(self.in3(self.deconv3(x))) + res
        x = self.deconv4(x)
        return x


class SynthesisPriorNet(nn.Module):
    """SynthesisPriorNet"""
    def __init__(self, out_channels_n=128, out_channels_m=192, withDLMM: bool = False):
        super(SynthesisPriorNet, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)

        self.deconv2 = nn.ConvTranspose2d(out_channels_n, out_channels_n, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)

        self.deconv3 = nn.ConvTranspose2d(out_channels_n, out_channels_m, kernel_size=3, stride=1, padding=1)

        self.withDLMM = withDLMM
        if withDLMM:
            self.conv_tail = nn.Conv2d(out_channels_m, 12 * out_channels_m, kernel_size=1, stride=1, padding=0)
            # 12 = 3 * 4, DLMM_Channels: Channels * 4 * len(["mu", "scale", "mix"])

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        x = self.deconv3(x)
        if self.withDLMM:
            self.conv_tail(x)
        # x = torch.exp(x)
        return x


class SynthesisPriorCANet(SynthesisPriorNet):
    """SynthesisPriorNet with channel attention"""
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
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_std = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)

        self.bit_estimator_z = BitEstimator(out_channels_n)

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

        quant_noise_feature = nn.init.uniform_(
            torch.zeros_like(quant_noise_feature), 
            -0.5, 0.5
        ).to(x.device)
        quant_noise_z = nn.init.uniform_(
            torch.zeros_like(quant_noise_z), 
            -0.5, 0.5
        ).to(x.device)

        # step 2: input image into encoder
        feature = self.encoder(x)
        batch_size = feature.shape[0]

        # step 3: put feature into prior(entropy encoding)
        z = self.encoder_prior(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)

        # step 4: compress the feature by the prior(entropy decoding)
        recon_mu = self.decoder_prior_mu(compressed_z)
        recon_sigma = self.decoder_prior_std(compressed_z)
        # recon_mu, recon_sigma = self.decoder_prior(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        compressed_feature_renorm = utils.quantize_st(compressed_feature_renorm, mean=recon_mu)

        # step 5: get the image from decoder
        recon_image = self.decoder(compressed_feature_renorm)
        # clipped_recon_image = recon_image.clamp(0, 255)       # because it's float, clamp it

        total_bits_feature, _ = utils.feature_probs_based_sigma(
            compressed_feature_renorm, recon_sigma
        )
        total_bits_z, _ = utils.iclr18_estimate_bits(self.bit_estimator_z, compressed_z)

        bpp_feature = total_bits_feature / (batch_size * x.shape[2] * x.shape[3])
        bpp_z = total_bits_z / (batch_size * x.shape[2] * x.shape[3])

        return recon_image, bpp_feature, bpp_z


# ---------
# HiFIC
# ---------
class HiFGenerator(nn.Module):
    def __init__(self, channels: int = 16, out_channels: int = 192, num_resblocks: int = 4, num_upblocks: int = 3):
        super().__init__()
        self.num_upblocks = num_upblocks
        self.num_resblocks = num_resblocks
        # channels: (1--> 16)
        self.conv_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=True)
        )

        # channels: (16 --> 32 --> 64 --> 128)
        for i in range(num_upblocks):
            resblock = ResBlock(channels * (2 ** i))
            self.add_module(f"res{str(i)}", resblock)
        for i in range(num_resblocks):
            resblock = ResBlock(channels * (2 ** (num_upblocks)), concat=False)
            self.add_module(f"res{str(i+num_upblocks)}", resblock)

        # channels: 128 shape: (64, 64) --> (512, 512)
        for i in range(num_upblocks):
            upconvblock = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=channels * (2 ** (num_resblocks - i - 1)), 
                    out_channels=channels * (2 ** (num_resblocks - i - 2)),
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(channels * (2 ** (num_resblocks))),
                nn.ReLU(inplace=True)
            )
            self.add_module(f"upconv{str(i)}", upconvblock)

        self.conv_tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1),
        )
    
    def forward(self, x):
        x = self.conv_head(x)
        for i in range(self.num_upblocks + self.num_resblocks):
            resblock = getattr(self, f'res{str(i)}')
            x = resblock(x)
            if i == self.num_upblocks:
                res = x
        x = x + res
        del res
        for i in range(self.num_upblocks):
            upblock = getattr(self, f'upconv{str(i)}')
            x = upblock(x)
        x = self.conv_tail(x)
        return x


class HiFEncoder(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = 192, num_downblocks: int = 3):
        super().__init__()
        self.num_downblocks = num_downblocks
        self.conv_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        for i in range(num_downblocks):
            downconvblock = nn.Sequential(
                nn.ReflectionPad2d((0, 1, 1, 0)),
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.add_module(f"downconv{str(i)}", downconvblock)

        self.conv_tail = nn.Sequential(
            ChannelNorm2d(channels),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1),
        )
    
    def forward(self, x):
        x = self.conv_head(x)
        for i in range(self.num_downblocks):
            downblock = getattr(self, f'downconv{str(i)}')
            x = downblock(x)
        x = self.conv_tail(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels: int = 64, num_convs: int = 4):
        super().__init__()
        self.context_conv = nn.Conv2d(1, channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.context_upsample = nn.Upsample(scale_factor=8, mode="nearest")
        for i in range(num_convs):
            if i == 0:
                in_channels: int = channels + 1
            else:
                in_channels: int = channels * (2 ** i)
            convblock = nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels, channels * (2 ** (i+1)), 
                    kernel_size=3, padding=1, padding_mode="reflect"
                )
            )
            self.add_module(f"conv{str(i)}", convblock)
        self.conv_tail = nn.Conv2d(channels * (2 ** num_convs), 1, kernel_size=1, stride=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x, y):
        """
        x: image after generator
        y: image
        """
        y = self.activation(self.context_conv(y))
        y = self.context_upsample(y)

        x = torch.cat([x, y], dim=1)
        x = self.activation(self.conv0(x))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        x = self.conv_tail(x).view(-1, 1)
        return x


class Hyperior(nn.Module):
    def __init__(self, out_channels_n=128, out_channels_m=192):
        super().__init__()
        self.analysis_net = AnalysisPriorNet(out_channels_n, out_channels_m)
        self.synthesis_net_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.synthesis_net_sigma = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.out_channels_n = out_channels_n
        self.out_channels_m = out_channels_m

    def forward(self, x):
        """
        x: features from encoder
        """
        # step 1: init
        quant_noise_feature = torch.zeros(
            x.size(0),
            self.out_channels_m, x.size(2), x.size(3)
        ).to(x.device)
        quant_noise_z = torch.zeros(
            x.size(0),
            self.out_channels_n, x.size(2) // 4, x.size(3) // 4
        ).to(x.device)

        quant_noise_feature = nn.init.uniform_(
            torch.zeros_like(quant_noise_feature),
            -0.5, 0.5
        ).to(x.device)
        quant_noise_z = nn.init.uniform_(
            torch.zeros_like(quant_noise_z),
            -0.5, 0.5
        ).to(x.device)

        # step2: entropy
        hyper_z = self.analysis_net(x)
        if self.training:
            hyper_z = hyper_z + quant_noise_z
        else:
            hyper_z = utils.quantize(hyper_z)

        bit_estimator = BitEstimator(channel=self.out_channels_n).to(x.device)
        hyper_z_bits, _ = utils.iclr18_estimate_bits(bit_estimator, hyper_z)
        hyper_z_bpp = hyper_z_bits / (x.shape[0] * x.shape[2] * x.shape[3])

        mu = self.synthesis_net_mu(hyper_z)
        sigma = self.synthesis_net_sigma(hyper_z)

        # maybe need gdn
        if self.training:
            hyper_features = x + quant_noise_feature
        else:
            hyper_features = utils.quantize(x, mean=mu)

        hyper_features_bits, _ = utils.feature_probs_based_sigma(hyper_features, sigma, mu)
        hyper_features_bpp = hyper_features_bits / (x.shape[0] * x.shape[2] * x.shape[3])

        features = utils.quantize_st(x, mean=mu)
        return hyper_z_bpp, hyper_features_bpp, features

class HiFImageCompression(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HiFEncoder()
        self.generator = HiFGenerator()
        self.hyperprior = Hyperior()

    def forward(self, x):
        features = self.encoder(x)
        z_bpp, feature_bpp, features = self.hyperprior(features)
        recon_image = self.generator(features)
        return recon_image, feature_bpp, z_bpp


if __name__ == "__main__":
    from torchsummary import summary
    a = HiFImageCompression().cuda()
    summary(a, (1, 64, 64))
    # if you use HiFIC, torchsummary may raise TypeError
