import os
import cv2
import numpy as np
import tfrecord


data_path = "flicker_2W_images"
size_image = 256
stride = 200

cnt = 0
writer = tfrecord.TFRecordWriter("train.tfrecord")
for img_name in os.listdir(data_path):
    cnt += 1
    print(img_name)
    img = cv2.imread(os.path.join(data_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for x in np.arange(0, img.shape[0] - size_image + 1, stride):
        for y in np.arange(0, img.shape[1] - size_image + 1, stride):
            # print("croping----")
            img_part = img[int(x): int(x + size_image),
                           int(y): int(y + size_image)]
            img_part = img_part.transpose(2, 0, 1)
            img_bytes = img_part.tobytes()

            writer.write({
                "image": (img_bytes, "byte"),
                "size": (size_image, "int"),
            })
    if cnt > 18000:
        break
writer.close()

cnt = 0
writer = tfrecord.TFRecordWriter("valid.tfrecord")
for img_name in os.listdir(data_path):
    cnt += 1
    print(img_name)
    if cnt < 18001:
        continue
    img = cv2.imread(os.path.join(data_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for x in np.arange(0, img.shape[0] - size_image + 1, stride):
        for y in np.arange(0, img.shape[1] - size_image + 1, stride):
            img_part = img[int(x): int(x + size_image),
                           int(y): int(y + size_image)]
            img_part = img_part.transpose(2, 0, 1)
            img_bytes = img_part.tobytes()

            writer.write({
                "image": (img_bytes, "byte"),
                "size": (size_image, "int"),
            })
writer.close()
