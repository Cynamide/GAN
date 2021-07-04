import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
SEED_SIZE = 100

noise = tf.random.normal([1,1,1, 128])

generator = tf.keras.models.load_model("YOUR FINAL MODEL PATH")

generated_image1 = generator(noise,training=True)
generated_image1 = ((generated_image1 + 1) * 0.5) * 255
generated_image1 = np.squeeze(generated_image1)

cv2.imwrite("img.png",cv2.cvtColor(generated_image1, cv2.COLOR_RGB2BGR))
