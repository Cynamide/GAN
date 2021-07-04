# Wasserstein GAN with Gradient Penalty (WGAN-GP) in TensorFlow
<p align="center"><img src="https://i.imgur.com/CIZr6Pw.gif"> </p>  

# Description

This is my [TensorFlow](https://www.tensorflow.org/) implementations of Wasserstein GANs with Gradient Penalty (WGAN-GP) proposed in [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028.pdf).The key insight of WGAN-GP is as follows. To enforce Lipschitz constraint in Wasserstein GAN, the original paper proposes to clip the weights of the discriminator (critic), which can lead to undesired behavior including exploding and vanishing gradients. Instead of weight clipping, this paper proposes to employ a gradient penalty term to constrain the gradient norm of the critic’s output with respect to its input, resulting the learning objective:
<p align="center">
    <img src="https://i.imgur.com/B9z5TQi.png" height="75"/>
</p>
This enables stable training of a variety of GAN models on a wide range of datasets.

## Prerequisites

- Python 3.7
- [Tensorflow 2.4.x](https://github.com/tensorflow/tensorflow/)
- [NumPy](http://www.numpy.org/)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)

## Running the Notebook
- This Notebook uses the Dataset obtained from [Kaggle](https://www.kaggle.com/soumikrakshit/anime-faces)
- Open up the train.ipynb file. Then make sure to  change the variables
	```bash
	PHOTO_PATH = "YOUR DATASET PATH HERE"
	MODEL_PATH = "MODEL PATH HERE"
	SAVE_PATH = 'MODEL SAVE PATH HERE'
	```
	to point to your desired Path folders. 
	
- Make sure to give the ```EPOCHS``` variable a higher number to get a more realistic result
- Now run the notebook

## Use the trained generator model to generate faces

- Open generate_face.py and change the variable 
	```bash
	MODEL_PATH = "YOUR FINAL MODEL PATH"
	```
	to point to your desired Model Path.
	
- Now run the file using :
	```bash
	python generate_face.py
	```
## Generate a video of the training process 

- Open generate_face.py and change the variable 
	```bash
	DATA_PATH = "MODEL FOLDER"
	```
	to point to your desired Model Folder.
- Now run the file using :
	```bash
	python generate_video.py
	```
