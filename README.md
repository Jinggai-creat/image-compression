# image-compression

image compression for `.nii`

# Usage

After downloading the data

> python make_dataset.py

you can get `train.tfrecord and valid.tfrecord`

> python -W ignore train.py --batch_size <your batch size> --niter <your iters>
