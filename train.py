import random
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from torch.autograd import Variable
from tfrecord.torch.dataset import TFRecordDataset

from tqdm import tqdm

import model
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
length = 206516

valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
valid_dataloader = dataloader.DataLoader(dataset=valid_dataset, batch_size=1)

# models init
model = model.EDICImageCompression().to(device)
model.apply(utils.weights_init)

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

            outputs, bpp = model(inputs)

            model_optimizer.zero_grad()

            loss = criterion(inputs, outputs)
            loss.backward()
            utils.clip_gradient(model_optimizer, 5)

            epoch_losses.update(loss.item(), len(inputs))

            model_optimizer.step()

            t.set_postfix(
                loss='{:.6f}'.format(epoch_losses.avg),
                bpp='{:.6f}'.format(bpp)
            )
            t.update(len(inputs))

    model_scheduler.step()

    # test
    model.eval()

    epoch_pnsr = utils.AverageMeter()
    epoch_ssim = utils.AverageMeter()

    for record in valid_dataloader:
        inputs = record["image"].reshape(
            1,
            3,
            record["size"][0],
            record["size"][0],
        ).float().to("cuda")

        with torch.no_grad():
            output, _ = model(inputs[0])
            epoch_pnsr.update(utils.calc_pnsr(output, inputs[0]), len(inputs))
            epoch_ssim.update(utils.calc_ssim(output, inputs[0]), len(inputs))

    print('eval psnr: {:.4f} eval ssim: {:.4f}'.format(epoch_pnsr.avg, epoch_ssim.avg))
