
Goal: This repository goes through my code for implementation of Dog vs Cat image classification challenge on Kaggle
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
This was my Capstone project for the Udacity machine learning nano degree

Softwares and Libraries required
•	Python 2.7 with Numpy, Scipy and Sklearn installed
•	Tensorflow (version >=0.9) See installation instructions at       https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md

•	TFLearn Installation: The easiest way is to run:
pip install tflearn
Or review installation instructions at http://tflearn.org/installation/

Input Dataset
The dataset for this project is too big to be zipped. Please download train.zip from Kaggle. See link below:
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
This dataset has 25000 images – 12,500 for cat and 12,500 for dogs. Please store in a directory and reference that directory in the final code attached with this project

Pickled Trained Model
The model for this project was trained on Amazon EC2 GPU 2.2x Large. It took about 8-10 minutes to train the model on GPU. Processing on CPU would be much slower. If you don’t have access to GPU to train the model, I have included the pickled train model in the zip file and alternate instructions in the code to simply load the pickled model and evaluate prediction accuracy on the test dataset.

Final code: final_code.py
Final project report - capstone_report.pdf

References
To succeed in this project I have leveraged several sources to build my understanding of Deep learning and also learn about implementation using TensorFlow. These sources include:
•	Udacity Course: Deep Learning by Vincent Vanhoucke (Google and Udacity).
•	TensorFlow and TFLearn
http://tflearn.org/getting_started
https://www.tensorflow.org/versions/r0.10/tutorials/index.html
•	Books on Deep Learning
o	Artificial Intelligence for Humans, Volume 3: Deep Learning and Neural Networks by Jeff Heaton
o	Getting started with TensorFlow by Giancarlo Zaccone
•	CS231n: Convolutional Neural Networks for Visual Recognition — Andrej Karpathy's Stanford CS class
http://cs231n.github.io/
•	Research papers on latest techniques in convolution neural networks
http://arxiv.org/pdf/1412.6806.pdf
