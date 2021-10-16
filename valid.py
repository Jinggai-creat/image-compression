import time
import os
import torch
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

import models
import utils


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
    model = models.EDICImageCompression().to("cuda")
    model_params = torch.load(model_name)
    model.load_state_dict(model_params)

    model.eval()
    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()

    t0 = time.clock()
    cnt = 0
    for record in valid_dataloader:
        cnt += 1
        inputs = record["image"].reshape(
            1,
            3,
            record["size"][0],
            record["size"][0],
        ).float().to("cuda")

        with torch.no_grad():
            output, bpp_feature_val, bpp_z_val = model(inputs)
            epoch_pnsr.update(utils.calc_psnr(output, inputs), len(inputs))
            epoch_ssim.update(utils.calc_ssim(output, inputs), len(inputs))
    t1 = time.clock()

    print('model name: {} eval psnr: {:.4f} eval ssim: {:.4f} valid_time: {:.4f}s \n'.format(
        model_name, epoch_pnsr.avg, epoch_ssim.avg, (t1-t0) / cnt
    ))
