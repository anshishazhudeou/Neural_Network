# Standard imports
import numpy as np
import Network as Network
import mnist_loader
import matplotlib.pylab as plt
import copy
# (b) Implement Derivative of Cosine Proximity
# Cosine Proximity
def CosineProximity(y, t):
    '''
        C = CosineProximity(y, t)

        Evaluates the average cosine proximity for the batch.

        Inputs:
          y is a batch of samples, with samples stored in rows
          t is a batch of targets

        Output:
          C is the average cosine proximity (cost)
    '''
    C = -np.sum(y * t, axis=1)
    C /= np.linalg.norm(y, axis=1)
    C /= np.linalg.norm(t, axis=1)
    return np.sum(C) / Network.NSamples(y)


# CosineProximity_p
def CosineProximity_p(y, t):
    '''
        dCdy = CosineProximity_p(y, t)

        Computes the gradient of the cosine proximity cost function.

        Inputs:
          y is a batch of samples, with samples stored in rows
          t is a batch of targets

        Output:
          dCdy is an array the same size as y, holding the derivative
               of the cost with respect to each element in y
    '''

    E = CosineProximity(y, t)
    y2norm = np.linalg.norm(y, axis=1)
    t2norm = np.linalg.norm(t, axis=1)
    dCdy = (np.divide(t.transpose(), (y2norm * t2norm).transpose())).transpose() * (-1) - np.divide(y.transpose(), (y2norm ** 2).transpose()).transpose() * E
    return dCdy

# Read in 10000 MNIST samples
train, validate, test = mnist_loader.load_data_wrapper()
train_in = np.array(train[0][:10000])

# Display some sample digit images
plt.figure(figsize=[15,4])
n_digits = 4
for n in range(n_digits):
    idx = np.random.randint(10000)
    plt.subplot(2,4,n+1)
    plt.imshow(np.reshape(train_in[idx], [28,28]), cmap='gray')
    plt.axis('off')

net = Network.Network()
net.AddLayer(Network.Layer(784))

net.AddLayer(Network.Layer(50, act = 'logistic'))
net.AddLayer(Network.Layer(784, act = 'identity'))
net.cost = CosineProximity
net.cost_p = CosineProximity_p


progress = net.SGD(train_in, train_in, batch_size=50, epochs=300, lrate=1.)

print('training cost ' + str(net.Evaluate(train_in, train_in)))


## (d) View Reconstructions
plt.figure(figsize=[15,4])
n_digits = 9
lastlayer = net.FeedForward(test[0][:10000])
# plt.subplot(2,9,1)
i=0
while i < 10:
    idx = np.random.randint(10000)
    plt.subplot(2,9,i+1)
    if(i == np.argmax(test[1][:10000][idx])):
        plt.imshow(np.reshape(lastlayer[idx], [28,28]), cmap='gray')
        plt.axis('off')
        i += 1