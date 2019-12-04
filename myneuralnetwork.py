from mymathlib import *

# Main neural network class with train and test functions
class NeuralNetwork(object):
    # Initialize network
    def __init__(self, dimensions, activation):
        self.dims = dimensions
        self.activation = activation
        self.data = None # (yet) Set to data object used for training, on export
        self.lossPerEpoch = None # (yet) Saved here on export
        self.exportState = None
        self.cost = MSE
        self.initializeParameters()
        self.numLayers = len(dimensions)

    # Initializes the weights and biases
    def initializeParameters(self):
        self.numTrainingIterations = 0
        self.b = self.initializeBiases()
        self.w = self.initializeWeights()

    # Resizes the network with the specified dimensions
    def resize(self, newDims):
        self.dims = newDims
        self.numLayers = len(newDims)
        self.initializeParameters()

    # Summary of network parameters
    def __repr__(self):
        return(f'Biases: {self.b},\n\n Weights: {self.w}')

    # Returns a string representation of the weights and biases of the network
    def getNetwork(self):
        ret = '\n'+'-'*8 + f'Network with dimension {self.dims}' + '-'*8+'\n\n'
        for layer in range(len(self.w)):
            ret += f'Layer {layer}:\nWeights:\n'
            ret += stringFormat2dList(self.w[layer]) + '\n'
            ret += 'Biases:\n'
            ret += stringFormat2dList(transpose(self.b[layer])) + '\n\n'
        ret += "\n\n"
        return ret

    # Initialize bias terms for each layer, first layer needs no bias term
    def initializeBiases(self):
        b = []
        for layerIndex in range(1, len(self.dims)):
            b.append([])
            for node in range(self.dims[layerIndex]):
                b[layerIndex - 1].append(random.gauss(0, 1))
            b[layerIndex - 1] = transpose(b[layerIndex - 1])
        return b

    # Initialize weights for each layer
    def initializeWeights(self):
        w = []
        # ith + 1 layer nodes correspond to rows of weight matrix
        # ith layer nodes correspond to columns of weight matrix
        for layerIndex in range(0, len(self.dims) - 1):
            rows = self.dims[layerIndex+1]
            cols = self.dims[layerIndex]
            wMat = makeGaussian2dList(rows, cols, 0, 1)
            w.append(wMat)
        return w

    # Implemented based on routine described in algorithm 6.3 (feedforward) in
    # Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
    def forwardPropagation(self, inputList):
        # Network input is just list of features
        x = inputList[:]
        # Iterate through all layers in the network
        for layerIndex in range(self.numLayers - 1):
            # Retrieve the current biases and weights from network object
            layerBiasVec = self.b[layerIndex]
            layerWeightMat = self.w[layerIndex]
            # Compute z:vec = W:mat X x:vec
            z = matProd(layerWeightMat, x)
            # Compute a = g(z + bias term)
            x = self.activation(addVectors(z, layerBiasVec))
        return x

    # Returns the number of correctly predicted test samples based on "winner
    # takes all" (final classification goes to highest output node)
    def test(self, data):
        results = [(self.forwardPropagation(x), y) for (x, y) in data]
        count = 0
        for predicted, actual in results:
            winningLabelIndex = None
            highestPercentage = -1
            for i in range(len(predicted)):
                if predicted[i][0] > highestPercentage:
                    highestPercentage = predicted[i][0]
                    winningLabelIndex = i
            # test against true label
            if actual[winningLabelIndex] == [1]:
                count += 1
        return count

    # Backpropagation and gradient descent algorithm implemented on routine described in
    # Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
    # Specifically, Chapter 6 (Deep Feedforward Networks) section 5.4 on
    # Back-Propagation Computation in Fully Connected MLP (Multilayer Perceptron)
    # and algorithms 6.3 (feedforward) for computing the activations of each layer
    # and 6.4 (backward computation) for computing the gradients on those activations.
    def train(self, data, iterations, alpha):
        self.numTrainingIterations += iterations
        if iterations < 0:
            alpha = -alpha
            iterations = abs(iterations)

        for iteration in range(iterations):
            random.shuffle(data)
            if (self.numTrainingIterations + iteration) % 100 == 0:
                print(f'Iteration {self.numTrainingIterations} training...', end = "")

            # Initialize matrices to hold weight and bias gradients
            weightGradient = []
            biasGradient = []
            for layerWeightMat in self.w:
                rows, cols = len(layerWeightMat), len(layerWeightMat[0])
                weightGradient.append(make2dList(rows, cols))
            for layerBiasVec in self.b:
                biasGradient.append(transpose([0]*len(layerBiasVec)))

            # Backprop
            for x, y in data:
                weightGradientChange = []
                biasGradientChange = []
                for layerWeightMat in self.w:
                    rows, cols = len(layerWeightMat), len(layerWeightMat[0])
                    weightGradientChange.append(make2dList(rows, cols))
                for layerBiasVec in self.b:
                    biasGradientChange.append(transpose([0]*len(layerBiasVec)))

                # 1. Set initial activation equal to just the input vector
                a = x[:]
                aMat = [a[:]]
                zMat = list() # z is hypothesis before activation function

                # 2. Propagate forwards to compute activations of all layers
                for layer in range(len(self.w)): 
                    w, b = self.w[layer], self.b[layer]
                    z = matProd(w, a)
                    zb = addVectors(z, b)
                    a = self.activation(zb)
                    zMat += [zb[:]]
                    aMat += [a[:]]

                # 3. Compute error of output layer L and store for use in computer errors
                #    of prior layers
                error = hadamardProd(self.cost(aMat[-1], y, order = 1),
                                             self.activation(zMat[-1], order = 1)) 
                biasGradientChange[-1] = error 
                weightGradientChange[-1] = matProd(error,
                                                           transpose(aMat[-2])) 

                # 4. Propagate backwards to compute errors of all layers L-1, L-2,..., 2
                #    and keep record of these results with weightGradientChange
                #    and biasGradientChange (accumulator variables for the error)
                for layer in range(2, self.numLayers):
                    z = zMat[-layer]
                    weightTimesError = matProd(transpose(self.w[-layer + 1]),
                                                       error) 
                    derivativeOfActivationAtZ = self.activation(z, order = 1) 
                    error = hadamardProd(weightTimesError,
                                                 derivativeOfActivationAtZ) 
                    biasGradientChange[-layer] = error
                    aT = transpose(aMat[-layer - 1])
                    weightGradientChange[-layer] = matProd(error, aT) 

                # 5. Apply biasGradientChange and weightGradientChange to
                #    biasGradient and weightGradient
                for layer in range(len(self.w)):
                    weightGradient[layer] = matrixSum(weightGradient[layer],
                                                              weightGradientChange[layer])
                    biasGradient[layer] = matrixSum(biasGradient[layer],
                                                            biasGradientChange[layer])

            # 6. Update weights and biases of network by subtracting the averaged
            #    partial derivatives of the cost function with respect to
            #    the parameters for the training batch multiplied by the
            #    learning rate.
            for layer in range(len(self.w)): 
                dWeights = multiplyMatrixByScalar(-alpha/len(data),
                                                          weightGradient[layer])
                newWeights = matrixSum(self.w[layer], dWeights)
                self.w[layer] = newWeights
                dBias = multiplyMatrixByScalar(-alpha/len(data),
                                                       biasGradient[layer])
                self.b[layer] = addVectors(self.b[layer], dBias)
            
            if (self.numTrainingIterations + iteration) % 100 == 0:
                print(f'tested with {self.test(data)} / {len(data)} correct.')
    
    # Returns a tuple with the maximum and minimum weight in the network
    def getMaxMinWeight(self):
        minWeight = 100
        maxWeight = -100
        for wMatrix in self.w:
            for wVec in wMatrix:
                for w in wVec:
                    minWeight = w if w < minWeight else minWeight
                    maxWeight = w if w > maxWeight else maxWeight
        return (minWeight, maxWeight)
