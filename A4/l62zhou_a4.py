# Standard imports
import numpy as np
import Network as Network
import mnist_loader
import matplotlib.pylab as plt
import copy


import re
text = open('origin_of_species.txt').read().lower()
chars = sorted(list(set(text)))
chars.insert(0, "\0") #Add newline character
vocab_size = len(chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
idx = [char_indices[c] for c in text]

# Let's simplify it by keeping only letters and spaces
filt_idx = []
for i in idx:
    if i<=24:
        filt_idx.append(2)
    elif i>24:
        filt_idx.append(i)
blah = ''.join([indices_char[f] for f in filt_idx])
text = re.sub(' +', ' ', blah)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Character set: '+''.join(chars)+' (first char is a space)')

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
idx = [char_indices[c] for c in text]

print('There are '+str(vocab_size)+' characters in our character set')

''.join(indices_char[i] for i in idx[:70])

def char2vec(c):
    v = np.zeros(vocab_size)
    v[char_indices[c]] = 1.
    return v

def index2vec(i):
    v = np.zeros(vocab_size)
    v[i] = 1.
    return v

def vec2index(v):
    i = np.argmax(v)
    return i

def vec2char(v):
    return indices_char[vec2index(v)]

'''Form the dataset in sentences'''
sentences_length = 10
sentences = []
next_chars = []
for i in range(0, 10000 - sentences_length + 1):
    sentences.append(idx[i: i + sentences_length]) #Assume a sentence is made of X characters
    next_chars.append(idx[i + 1: i + sentences_length + 1]) #Offset by 1 to the right for the target

sentences = np.concatenate([[np.array(o)] for o in sentences[:-2]])
next_chars = np.concatenate([[np.array(o)] for o in next_chars[:-2]])
sentences.shape, next_chars.shape

def read_sentence(idx):
    return ''.join(indices_char[i] for i in idx)

print('Here is how you can view one of the samples:')
print('Sample input: ['+read_sentence(sentences[0])+']')


## Some utility functions

def sigma(z):
    return np.clip(z, a_min=0, a_max=None)  # ReLU
    #return 1./(1+np.exp(-z))  # use this for logistic

def sigma_primed(y):
    return np.clip(np.sign(y), a_min=0, a_max=1)  # Derivative of ReLU
    #return y*(1.-y)  # use this for logistic

def softmax(z):
    ez = np.exp(z)
    denom = np.sum(ez)
    return ez / denom

def CrossEntropy(y, t):
    return -sum(t*np.log(y))

## (a) Complete BPTT

class RNN():

    def __init__(self, dims, seq_length=10):
        '''
         Input:
           dims is [X, H, Y], where the input has layer has X neurons, the
                hidden layer has H neurons, and the output layer has Y neurons.
           seq_length is how many steps to unroll the RNN through time
                (this is the same as tau in the lecture notes)
        '''
        self.X, self.H, self.Y = dims
        self.seq_length = seq_length
        # Input layer
        self.xs = [np.zeros(self.X) for x in range(seq_length)]  # activity
        # Hidden layer
        self.hs = [np.zeros(self.H) for x in range(seq_length)]  # activity
        # Output layer
        self.ys = [np.zeros(self.Y) for x in range(seq_length)]  # activity

        # Connection weights
        self.U = np.random.normal(size=[self.H, self.X]) / np.sqrt(self.X)  # input->hidden
        self.W = np.random.normal(size=[self.H, self.H]) / np.sqrt(self.H)  # hidden->hidden
        self.V = np.random.normal(size=[self.Y, self.H]) / np.sqrt(self.H)  # hidden->output
        self.b = np.zeros(self.H)  # biases for hidden nodes
        self.c = np.zeros(self.Y)  # biases for output nodes

    def ForwardTT(self, seq_in):
        '''
         i = ForwardTT(seq_in)

         Propagates the RNN forward through time, saving all the intermediate
         states that will be needed for backprop through time (BPTT).

         Input:
           seq_in is a vector of indecies, with self.seq_length elements.

         Output:
           i is the index of the character predicted to follow the input.

         This method's main purpose is to update the states of the activites
         in the time-unrolled network.
        '''
        self.xs[0] = index2vec(seq_in[0])  # convert to character vector

        # Starting input current for hidden nodes
        ss = np.dot(self.U, self.xs[0]) + self.b
        self.hs[0] = sigma(ss)  # Activation of hidden nodes

        # Input current for output nodes
        zs = np.dot(self.V, self.hs[0]) + self.c
        self.ys[0] = softmax(zs)  # Activation of output nodes

        # Now process forward in time
        for i in range(1, self.seq_length):
            self.xs[i] = index2vec(seq_in[i])  # input vector

            # Input current for hidden nodes, including recurrent connections
            ss = np.dot(self.U, self.xs[i]) + np.dot(self.W, self.hs[i - 1]) + self.b
            self.hs[i] = sigma(ss)  # Activation

            # Input current for output nodes
            zs = np.dot(self.V, self.hs[i]) + self.c
            self.ys[i] = softmax(zs)  # Activation

        # Might as well output the final state of the output
        return vec2index(self.ys[-1])

    def BPTT(self, seq_in, seq_out):
        '''
         BPTT(seq_in, seq_out)

         Performs backprop through time on one sample given by the input and
         output sequence.

         Input:
           seq_in is a vector of indices specifying the input sequence of
                   characters.
           seq_out is a vector of indices specifying the output sequence of
                   characters. Typically, seq_out is the same as seq_in, but
                   shifted by 1 character.

         Output:
           None, but the connection weights and biases are updated.
        '''
        # Initialize gradients to zero
        self.dEdV = np.zeros(np.shape(self.V))
        self.dEdW = np.zeros(np.shape(self.W))
        self.dEdU = np.zeros(np.shape(self.U))
        self.dEdb = np.zeros(np.shape(self.b))
        self.dEdc = np.zeros(np.shape(self.c))

        # ===================
        # ===================
        # =  YOUR CODE HERE =
        # ===================
        # ===================

    def Generate(self, n=1):
        '''
         c = Generate(n=1)

         Runs the RNN from the last state after running ForwardTT, outputting
         the next n characters.

         Input:
           n is the number of characters you want to predict

         Output:
           c is a string of n characters
        '''
        y = self.ys[-1]  # Final output of ForwardTT
        c = vec2char(y)  # Convert it to a character string
        h = self.hs[-1]  # Starting with last hidden state...
        # Loop forward in time
        # (no need to record states, since we won't be doing BPTT)
        for nn in range(n - 1):
            x = copy.copy(y)  # Use last output as next input

            # Input current for hidden nodes
            s = np.dot(self.U, x) + np.dot(self.W, h) + self.b
            h = sigma(s)  # Activation

            # Input current for output nodes
            z = np.dot(self.V, h) + self.c
            y = softmax(z)  # Activation

            # And add the next character to our output string
            c += vec2char(y)

        return c

    def Evaluate(self, train_in, train_out):
        '''
         loss = Evaluate(train_in, train_out)

         Evaluates the network on the supplied dataset.

         Input:
           train_in is a list of input sequences (see ForwardTT for format of input)
           train_out is the corresponding list of output sequences

         Output:
           loss is the average cross entropy
        '''
        val = 0.
        for x, t in zip(train_in, train_out):
            self.ForwardTT(x)
            for i in range(self.seq_length):
                val += CrossEntropy(self.ys[i], index2vec(t[i]))
        return val / len(train_in)

    def Train(self, train_in, train_out, kappa=0.05, epochs=1):
        '''
         Train(train_in, train_out, kappa=0.05, epochs=1)

         Performs epochs of gradient descent, performing BPTT after each sample.

         Input:
           train_in and train_out is the training dataset
           kappa is the learning rate
           epochs is the number of times to go through the dataset

         Output:
           None, but the connection weights and biases are updated
        '''
        # Loop over epochs
        for e in range(epochs):

            # Shuffle the training data
            data_shuffled = list(zip(train_in, train_out))
            np.random.shuffle(data_shuffled)

            for x, t in data_shuffled:
                self.ForwardTT(x)  # Forward through time
                self.BPTT(x, t)  # Backprop through time
                # Note that BPTT starts by resetting the gradients to zero.

                # Apply update to connection weights and biases
                self.V -= kappa * self.dEdV
                self.U -= kappa * self.dEdU
                self.W -= kappa * self.dEdW
                self.b -= kappa * self.dEdb
                self.c -= kappa * self.dEdc

            print('Epoch ' + str(e) + ', Loss = ' + str(self.Evaluate(train_in, train_out)))


## (b) Create the RNN

# YOUR CODE HERE
#net = RNN(...)









## (c) Train

# YOUR CODE HERE
# net.Train(...)

# You might opt to have more than one train command, using different
# learning rates and numbers of epochs. Each one builds on the results
# from the last run.
#net.Train(...)




## (d) Evaluate
# A sample continuation.
n = 38
b.ForwardTT(sentences[n])
blah = read_sentence(sentences[n])
print('Input:      '+blah)
print('Prediction: '+blah+b.Generate(5))
print('Actual:     '+blah+read_sentence(sentences[n+10]))



blah = 'harles dar'
x = [char_indices[c] for c in blah]
b.ForwardTT(x)
print(blah)
print(blah+b.Generate(10))


# YOUR CODE HERE