import numpy as np
from PIL import Image
from tqdm import tqdm
import os

DATA_PATH = 'ADD YOUR DATAFILE PATH' 
img_width = 64
img_height = 64
channels = 3

training_data = []
for filename in tqdm(os.listdir(DATA_PATH)):
    path = os.path.join(DATA_PATH,filename)
    image = Image.open(path).resize((img_width,
            img_height),Image.ANTIALIAS)
    training_data.append(np.asarray(image))
training_data = np.reshape(training_data,(-1,img_width,
            img_height,channels))
training_data = training_data.astype(np.float32)
training_data = training_data / 127.5 - 1
print(np.shape(training_data))
np.save("training.npy",training_data)