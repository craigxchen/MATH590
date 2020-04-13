import network.network as Network
import network.mnist_loader as mnist_loader
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from ripser import ripser
from persim import plot_diagrams

with open('network/trained_network.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    net = u.load()
    
train_data, train_labels, test_data, test_labels = mnist_loader.load_data_wrapper()
    
def predict(n):
    # Get the data from the test set
    x = test_data[n]

    # Print the prediction of the network
    print('Network output: \n' + str(np.round(net.feedforward(x), 2)) + '\n')
    print('Network prediction: ' + str(np.argmax(net.feedforward(x))) + '\n')
    print('Actual image: ')
    
    # Draw the image
    plt.figure()
    plt.imshow(x.reshape((28,28)), cmap='Greys')
    plt.show()

# Replace the argument with any number between 0 and 9999
#predict(8384)


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
                                                                                                                                                                                
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def input_derivative(net, x, y):
    """ Calculate derivatives wrt the inputs"""
    nabla_b = [np.zeros(b.shape) for b in net.biases]
    nabla_w = [np.zeros(w.shape) for w in net.weights]
    
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(net.biases, net.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
        
    # backward pass
    delta = net.cost_derivative(activations[-1], y) * \
        sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, net.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(net.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
    # Return derivatives WRT to input
    return net.weights[0].T.dot(delta)

# %% non-targeted
def adversarial(net, n, steps, eta):
    """
    net : network object
        neural network instance to use
    n : integer
        our goal label (just an int, the function transforms it into a one-hot vector)
    steps : integer
        number of steps for gradient descent
    eta : float
        step size for gradient descent
    """
    # Set the goal output
    goal = np.zeros((10, 1))
    goal[n] = 1

    # Create a random image to initialize gradient descent with
    x = np.random.normal(.5, .3, (784, 1))

    # Gradient descent on the input
    for i in range(steps):
        # Calculate the derivative
        d = input_derivative(net,x,goal)
        
        # The GD update on x
        x -= eta * d
        
    return x

# Wrapper function
def generate(n):
    """
    n : integer
        goal label (not a one hot vector)
    """
    a = adversarial(net, n, 1000, 1)
    x = np.round(net.feedforward(a), 2)
    
    print('Network Output: \n' + str(x) + '\n')
    
    print('Network Prediction: ' + str(np.argmax(x)) + '\n')
    
    print('Adversarial Example: ')
    plt.figure()
    plt.imshow(a.reshape(28,28), cmap='Greys')
    plt.show()
    
    return a

adv_ex = generate(2)
adv_ex = adv_ex.reshape(28,28)

adv_dgms = ripser(adv_ex)['dgms']

print("Homology of Adversarial Example: ")

plt.figure()
plot_diagrams(adv_dgms, plot_only=[0], ax=plt.subplot(121))
plot_diagrams(adv_dgms, plot_only=[1], ax=plt.subplot(122))
plt.show()

# 3D Height Plots
xx, yy = np.mgrid[0:adv_ex.shape[0], 0:adv_ex.shape[1]]

fig = plt.figure()
ax1 = Axes3D(fig)
ax1.plot_surface(xx, yy, adv_ex ,rstride=1, cstride=1, cmap=plt.cm.gray,
        linewidth=0)

plt.show()