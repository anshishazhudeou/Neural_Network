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
        dEdz = []  # zichuang
        ## Note that we don't have sigma prime here because Porf Orchard said on piazza that it was simply (y-t)
        for i in range(self.seq_length):
            # each node in output layer -
            dEdz.append(self.ys[i] - index2vec(seq_out[i]))
        dEds = [0] * self.seq_length
        dEds[-1] = sigma_primed(self.hs[-1]) * (self.V.T @ dEdz[-1])
        for i in range(self.seq_length - 2, -1, -1):
            dEds[i] = sigma_primed(self.hs[i]) * ((self.V.T @ dEdz[i]) + (self.W.T @ dEds[i + 1]))
        for i in range(self.seq_length):
            self.dEdb += dEds[i]
            self.dEdc += dEdz[i]
            self.dEdV += np.array([dEdz[i]]).T @ np.array([self.hs[i]])
            self.dEdU += np.array([dEds[i]]).T @ np.array([self.xs[i]])
            if (i != self.seq_length - 1):
                self.dEdW += np.array([dEds[i + 1]]).T @ np.array([self.hs[i]])

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
net = RNN(dims = [27, 400, 27])
net.Train(sentences, next_chars, kappa = 0.001, epochs = 15)
print("shit")