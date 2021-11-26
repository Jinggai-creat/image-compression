# image-compression

image compression for `.nii.gz` / `.DICOM`

## Usage

After downloading the data, please make sure there is a dictionary called "dataset" contained `.nii.gz` files.

> python make_dataset.py

you can get `train.tfrecord and valid.tfrecord`

> python -W ignore train.py --batch_size <your batch size> --niter <your iters>

## quantize

use vitis-ai to quantize the model

metrics:

> PSNR: 33.613258361816406
> SSIM: 0.9811062216758728

about data size: JPEG2000: 5.7kb / ours: 2.9kb
