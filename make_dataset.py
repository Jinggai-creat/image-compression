import os
import sys

import cv2
import numpy as np
import nibabel as nib
import tfrecord


size_image = 64

for split in ["train", "valid"]:
    cnt = 0
    writer = tfrecord.TFRecordWriter("{}.tfrecord".format(split))
    data_path = "dataset/{}".format(split)
    for file_name in os.listdir(data_path):
        img = nib.load(os.path.join(data_path, file_name))
        print(file_name, img.header["db_name"])
        data = img.get_fdata()
        w, h, q, t = data.shape
        m = np.max(data)
        data = data / m * 255
        for x in range(q):
            for y in range(t):
                cnt += 1
                data_part = data[..., x, y].tobytes()
                writer.write({
                    "image": (data_part, "byte"),
                    "size": (size_image, "int"),
                })
    writer.close()
    print("length of " + split + ": {}".format(cnt))
