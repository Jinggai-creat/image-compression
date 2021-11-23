import time
import cv2
import os
import torch
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

from models_converted import *
import utils


class EDICImageCompressionValid(nn.Module):
    """EDICImageCompression for valid"""

    def __init__(self, out_channels_n=128, out_channels_m=192):
        super(EDICImageCompressionValid, self).__init__()
        self.out_channels_n = out_channels_n
        self.out_channels_m = out_channels_m

        self.encoder = AnalysisNet(out_channels_n, out_channels_m)
        self.decoder = SynthesisNet(out_channels_n, out_channels_m)

        self.encoder_prior = AnalysisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_mu = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior_std = SynthesisPriorNet(out_channels_n, out_channels_m)
        self.decoder_prior = SynthesisPriorCANet(out_channels_n, out_channels_m)

        self.bit_estimator_z = BitEstimator(out_channels_n)
    
    def forward(self, x):
        feature = self.encoder(x)
        recon_image = self.decoder(feature)

        z = self.encoder_prior(feature)
        recon_sigma = self.decoder_prior_std(z)
        total_bits_feature, _ = utils.feature_probs_based_sigma(
            feature, recon_sigma
        )
        bpp_feature = total_bits_feature / (x.shape[0] * x.shape[2] * x.shape[3])
        total_bits_z, _ = utils.iclr18_estimate_bits(self.bit_estimator_z, z)
        bpp_z = total_bits_z / (x.shape[0] * x.shape[2] * x.shape[3])
        return recon_image, bpp_feature, bpp_z


for name in os.listdir("model"):
    model_name = os.path.join("model", name)
    # dataset init, train file need .tfrecord
    description = {
        "image": "byte",
        "size": "int",
    }
    valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
    valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

    # models init
    model = EDICImageCompressionValid().to("cuda")
    model_params = torch.load(model_name)
    model.load_state_dict(model_params)

    model.eval()
    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()
    epoch_bpp = utils.AverageMeter()

    t0 = time.clock()
    cnt = 0
    for record in valid_dataloader:
        cnt += 1
        inputs = record["image"].reshape(
            1,
            1,
            record["size"][0],
            record["size"][0],
        ).float().to("cuda")

        cv2.imshow("input", inputs.cpu().squeeze().numpy()/255)

        with torch.no_grad():
            output, bpp_feature_val, bpp_z_val = model(inputs)
            epoch_pnsr.update(utils.calc_psnr(output, inputs), len(inputs))
            epoch_ssim.update(utils.calc_msssim(output, inputs), len(inputs))
            epoch_bpp.update((bpp_feature_val + bpp_z_val), len(inputs))
        cv2.imshow("bpp{}".format(bpp_feature_val), output.cpu().squeeze().numpy()/255)
        cv2.waitKey()
        cv2.destroyAllWindows()
        if cnt == 10:
            break
    t1 = time.clock()

    print('model name: {} eval psnr: {:.4f} eval ssim: {:.4f} eval bpp: {:.4f} valid_time: {:.4f}s \n'.format(
        model_name, epoch_pnsr.avg, epoch_ssim.avg, epoch_bpp.avg, (t1-t0) / cnt
    ))
