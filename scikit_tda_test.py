from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

#np.random.seed(5)
RAND = np.random.randint(50001)

with np.load('mnist.npz') as data: 
    # data is stored in 28 by 28 images (784 brightness values)
    # training data and labels will be stored in 2D arrays - there are 50000 training samples, thus .shape = (50000,784)
    training_images = data['training_images']
    training_labels = data['training_labels']

training_images = training_images.reshape(50000,28,28)

dgms = ripser(training_images[RAND])['dgms']
plot_diagrams(dgms, show=True)

plot_diagrams(dgms, plot_only=[0], ax=plt.subplot(121))
plot_diagrams(dgms, plot_only=[1], ax=plt.subplot(122))

fig, ax = plt.subplots()
plt.imshow(training_images[RAND])
plt.show()