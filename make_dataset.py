import os
import sys

import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import tfrecord


size_image = 64


def clip(file_path):
    image_paths = file_path.split(".")[0]
    if os.path.exists(image_paths):
        return image_paths
    os.makedirs(image_paths)
    image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image)
    for i in range(data.shape[0]):
        out = sitk.GetImageFromArray(data[i, ...])
        sitk.WriteImage(out, os.path.join(image_paths, "{}.nii.gz".format(i+1)))
    return image_paths


def norm(file_path, maximum: int = 255, minimum: int = 0):
    image = sitk.ReadImage(file_path)
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = resacleFilter.Execute(image)
    data = sitk.GetArrayFromImage(image)
    return data


for split in ["train", "valid"]:
    cnt = 0
    writer = tfrecord.TFRecordWriter("{}.tfrecord".format(split))
    data_path = "dataset/{}".format(split)
    for file_name in os.listdir(data_path):
        if ".nii.gz" not in file_name:
            continue
        file_path = clip(os.path.join(data_path, file_name))
        print("saved 3d image in: {} ".format(file_path))
        for image_name in os.listdir(file_path):
            data = norm(os.path.join(file_path, image_name))
            for y in range(data.shape[0]):
                cnt += 1
                data_part = data[y, ...].astype(np.uint8)
                assert np.isnan(data_part).sum() == 0 and np.isinf(data_part).sum() == 0
                writer.write({
                    "image": (data_part.tobytes(), "byte"),
                    "size": (size_image, "int"),
                })
    writer.close()
    print("length of " + split + ": {}".format(cnt))
