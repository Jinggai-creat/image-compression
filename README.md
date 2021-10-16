# image-compression

image compression for `.nii.gz`

# Usage

After downloading the data, please make sure there is a dictionary called "dataset" contained `.nii.gz` files.

> python make_dataset.py

you can get `train.tfrecord and valid.tfrecord`

> python -W ignore train.py --batch_size <your batch size> --niter <your iters>
