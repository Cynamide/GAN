import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os 

frameSize = (64, 64)
out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'mp4v'), 3, frameSize)
noise = tf.random.normal([1,1,1, 128])
DATA_PATH = "MODEL FOLDER"
for i in tqdm(range(701)):
    if i == 100:
        pass
    elif i%20 == 0:
        path = os.path.join(DATA_PATH,f"face_generator{i}.h5")
        generator = tf.keras.models.load_model(path)
        generated_image = generator(noise, training = False)
        proto_tensor = tf.make_tensor_proto(generated_image)
        generated_image = tf.make_ndarray(proto_tensor)
        generated_image = (generated_image + 1) * 127.5
        generated_image = generated_image[0].astype('uint8')
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)
        out.write(generated_image)

out.release()


