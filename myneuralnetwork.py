# pylint: disable=no-member

# This file houses the neural network class with an implementation of the
# gradient descent and backpropagation algorithms.
from mymathlib import *
from helpers112 import stringFormat2dList

import numpy as np

# The main Neural Network class
class NeuralNetwork(object):
    # Initialize network
    def __init__(self, dimensions, activation):
        self.dims = dimensions
        self.activation = activation
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

    # Initialize bias terms for each layer, first layer takes no bias term
    def initializeBiases(self):
        b = []
        for layerIndex in range(1, len(self.dims)):
            b.append(makeGaussian2dList(self.dims[layerIndex], 1, 0, 1))
            '''
            b.append([])
            for node in range(self.dims[layerIndex]):
                b[layerIndex - 1].append(random.gauss(0, 1))
            b[layerIndex - 1] = transpose(b[layerIndex - 1])
            '''
        return b

    # Initialize weights for each layer
    def initializeWeights(self):
        w = []
        # ith + 1 layer nodes correspond to rows of weight matrix
        # ith layer nodes correspond to columns of weight matrix
        for layerIndex in range(0, len(self.dims) - 1):
            # i-th row of interlayer weight matrix corresponds to i-th node in proceeding layer
            rows = self.dims[layerIndex+1]
            # j-th column of interlayer weight matrix corresponds to j-th node in current layer
            cols = self.dims[layerIndex]
            wMat = makeGaussian2dList(rows, cols, 0, 1)
            w.append(wMat)
        return w

    # Implemented based on routine described in algorithm 6.3 (feedforward) in
    # Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
    def forwardPropagation(self, inputList):
        # Network input is just list of features
        activation = inputList[:]
        # Iterate through all layers in the network
        for layerIndex in range(self.numLayers - 1):
            # Retrieve the current biases and weights from network object
            layerBiasVec = self.b[layerIndex]
            layerWeightMat = self.w[layerIndex]
            # Compute z:vec = W:mat X x:vec
            z = matProd(layerWeightMat, activation)
            # Compute a = g(z + bias term)
            activation = self.activation(addVectors(z, layerBiasVec))
            '''
            # Retrieve the current biases and weights from network object
            layerBiasVec = self.b[layerIndex]
            layerWeightMat = self.w[layerIndex]
            # Compute z:vec = W:mat X x:vec
            z = np.dot(layerweightMat, activation)
            # Compute a = g(z + bias term)
            activation = self.activation(z + layerBiasVec)
            '''
        return activation

    # Returns the number of correctly predicted test samples based on "winner
    # takes all" (final classification goes to highest output node)
    def testClassificationAccuracy(self, data):
        results = [(self.forwardPropagation(x), y) for (x, y) in data]
        count = 0
        for predicted, actual in results:
            winningLabelIndex = None
            highestPercentage = -1
            for i in range(len(predicted)):
                if predicted[i][0] > highestPercentage:
                    highestPercentage = predicted[i][0]
                    winningLabelIndex = i
            # Test against true label
            if actual[winningLabelIndex] == [1]:
                count += 1
        return count

    # Backpropagation and gradient descent implemented as described in
    # Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT
    # press, 2016. Chapter 6 (Deep Feedforward Networks) section 5.4 on
    # Back-Propagation Computation in Fully Connected MLP (Multilayer Perceptron)
    # and algorithms 6.3 (feedforward) for computing the activations of each
    # layer and 6.4 (backward computation) for computing the gradients on those
    # activations.
    def train(self, data, iterations, alpha):
        self.numTrainingIterations += iterations
        for _ in range(iterations):
            random.shuffle(data)
            # Initialize matrices to hold weight and bias gradients
            weightGradient = []
            biasGradient = []
            for layerWeightMat, layerBiasVec in zip(self.w, self.b):
                rows, cols = len(layerWeightMat), len(layerWeightMat[0])
                weightGradient.append(make2dList(rows, cols))
                rows = len(layerBiasVec)
                biasGradient.append(make2dList(rows, 1))

            # Backpropagation Algorithm
            for x, y in data:
                weightGradientChange = []
                biasGradientChange = []
                for layerWeightMat, layerBiasVec in zip(self.w, self.b):
                    rows, cols = len(layerWeightMat), len(layerWeightMat[0])
                    weightGradientChange.append(make2dList(rows, cols))
                    rows = len(layerBiasVec)
                    biasGradientChange.append(make2dList(rows, 1))

                # 1. Set initial activation equal to just the input vector
                a = x[:]
                aMat = [a[:]]
                zbMat = list() # z is hypothesis before activation function

                # 2. Propagate forwards to compute activations of all layers
                for layer in range(len(self.w)):
                    w, b = self.w[layer], self.b[layer]
                    z = matProd(w, a)
                    zb = addVectors(z, b)
                    a = self.activation(zb)
                    zbMat += [zb[:]]
                    aMat += [a[:]]

                # 3. Compute error of output layer L and store for computing
                #    errors of prior layers
                derivativeOfCostFunctionAtActivation = self.cost(aMat[-1], y, order = 1)
                derivativeOfActivationFunctionAtZb = self.activation(zbMat[-1], order = 1).transpose()
                error = hadamardProd(derivativeOfCostFunctionAtActivation,
                                     derivativeOfActivationFunctionAtZb).transpose()
                biasGradientChange[-1] = error
                weightGradientChange[-1] = matProd(error, transpose(aMat[-2])) 

                # 4. Propagate backwards to compute errors of all layers
                #    L-1, L-2,..., 2 and keep record of these results with
                #    weightGradientChange and biasGradientChange (accumulator
                #    variables for the error)
                for layer in range(2, self.numLayers):
                    zb = zbMat[-layer]
                    weightTimesError = matProd(transpose(self.w[-layer + 1]),
                                               error) 
                    derivativeOfActivationAtZb = self.activation(zb, order = 1) 
                    error = hadamardProd(weightTimesError,
                                         derivativeOfActivationAtZb) 
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

            # 6. Update weights and biases of network by subtracting the
            #    averaged partial derivatives of the cost function with respect
            #    to the parameters for the training batch multiplied by the
            #    learning rate.
            for layer in range(len(self.w)): 
                dWeights = multiplyMatrixByScalar(-alpha/len(data),
                                                          weightGradient[layer])
                newWeights = matrixSum(self.w[layer], dWeights)
                self.w[layer] = newWeights
                dBias = multiplyMatrixByScalar(-alpha/len(data),
                                                       biasGradient[layer])
                self.b[layer] = addVectors(self.b[layer], dBias)