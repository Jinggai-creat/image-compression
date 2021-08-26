import random
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

import models
import config
import utils


opt = config.get_options()

# deveice init
CUDA_ENABLE = torch.cuda.is_available()
device = torch.device('cuda:0' if CUDA_ENABLE else 'cpu')

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init, train file need .tfrecord
description = {
    "image": "byte",
    "size": "int",
}
train_dataset = TFRecordDataset("train.tfrecord", None, description)
# do not shuffle
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
# TODO(hujiakui): too slow when batch_size is 4, length will be 206516
# length = utils.get_length(train_dataloader)
length = 825856

valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

# models init
model = models.EDICImageCompression().to(device)

# criterion init
criterion = torch.nn.MSELoss()

# optim and scheduler init
model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

# train model
print("-----------------train-----------------")
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()

    with tqdm(total=(length - length % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for record in train_dataloader:
            inputs = record["image"].reshape(
                opt.batch_size,
                3,
                record["size"][0],
                record["size"][0],
            ).float().to("cuda")

            outputs, bpp_feature, bpp_z = model(inputs)

            model_optimizer.zero_grad()

            # train lambda: 8192, add msssim-loss
            print(utils.calc_msssim(inputs, outputs))
            loss = criterion(inputs, outputs) + \
                   (bpp_feature + bpp_z) * 8192 - \
                   20 * torch.log(utils.calc_msssim(inputs, outputs) + 1e-7) * (epoch+1)
            loss.backward()
            utils.clip_gradient(model_optimizer, 5)

            model_optimizer.step()
            epoch_losses.update(loss.item(), len(inputs))

            t.set_postfix(
                loss='{:.6f}'.format(epoch_losses.avg),
                bpp_feature='{:.6f}'.format(bpp_feature),
                bpp_z='{:.6f}'.format(bpp_z)
            )
            t.update(len(inputs))

    model_scheduler.step()

    # test, just pick one to take a look
    model.eval()
    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()

    cnt = 0
    for record in valid_dataloader:
        cnt += 1
        if cnt >= 100:
            break
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

    print('eval psnr: {:.4f} eval ssim: {:.4f}\n'.format(epoch_pnsr.avg, epoch_ssim.avg))
    torch.save(model.state_dict(), "edic_epoch_{}_bpp_{}.pth".format(epoch, bpp_feature_val+bpp_z_val))
