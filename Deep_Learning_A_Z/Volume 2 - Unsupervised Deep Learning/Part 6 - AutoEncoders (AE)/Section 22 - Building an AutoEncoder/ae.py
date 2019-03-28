# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset                                                                 # since some moive names contain some special char, so we cannot use classic enconding
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users] # contain movie_id rated by a specific user
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies) # create np array with nb_movies zeros
        ratings[id_movies - 1] = id_ratings # ratings is a list, so we need to add an offset
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ): # the architecture of neural network
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # full connect bet the first input vector features and the first encoded vector, 20 is an experimental value
        self.fc2 = nn.Linear(20, 10) # second hidden layer
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x): # x is the input vector
        x = self.activation(self.fc1(x)) # first encoded vector
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # output vector so we dont have to apply encoding part
        return x
sae = SAE()
criterion = nn.MSELoss()
# the decay is used to reduce the learning rate after every few epochs and that's
# in dorer to regulate the convergence
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # stochastic descent optimization

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0. # the number of users who update at least one movive.
    for id_user in range(nb_users): # in one epoch
        input = Variable(training_set[id_user]).unsqueeze(0) #create new dimension for pytorch
        target = input.clone()
                #rate of an user
        if torch.sum(target.data > 0) > 0: # optimize memory
            output = sae(input) # predicted rating
            target.require_grad = False # ensure we dont compute the gradient with respect to target
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) #/number of moives which do have postive rating
            loss.backward() # decide the diretion that weight needs to be update
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step() # decide intensity/amount that weight needs to be updated
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss  /s))