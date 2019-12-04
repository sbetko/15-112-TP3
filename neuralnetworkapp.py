import math, random, copy, numbers, os, decimal

# Taken from course site: https://www.cs.cmu.edu/~112/notes/cmu_112_graphics.py
from cmu_112_graphics import *
from tkinter import *
# Helper functions adapted from course website
from helpers112 import *

# My imports and code
from mydatasetlib import Dataset
from myneuralnetwork import NeuralNetwork
from mymathlib import *

# This is for performance purposes
import psutil
psutil.Process(os.getpid()).nice(psutil.REALTIME_PRIORITY_CLASS)

# Things To Do:
# DONE: Add CSV read-support
# DONE: Add a graph for the loss function with matplotlib
# TODO: Implement Stochastic Gradient Descent
#       - With variable batch size too
# DONE Add mouse-hover to view parameter
# TODO: Add validation set
# TODO: Add a TEST MODE
# TODO: Add model import / export
# TODO: Add input/output ghost for config help
# TODO: Add a training set generator with choice of simple and complex boolean
#       rules, with the respective test set generator
# TODO: Add a training set generator with choice of mathematical function on a
#       chosen domain :) (with respective test set generator)
# TODO: Add a graph for the decision boundary! (with points plotted)
# TODO: Add a mouse-based GUI with TkInter
# TODO: Add ability to view backpropagation process
#       - Step, layer by layer, and view change to weight
# TODO: Add support for regression
# TODO: Add a plot of feature space? Only works for 2d, not really helpful
# TODO: Speed up the matrix math a little bit

class startMode(Mode):
    def appStarted(self):
        pass

    def mousePressed(self, event):
        pass
    
    def redrawAll(self, canvas):
        title = "Build Your Own Neural Network"
        titleFont = "Helvetica 26"
        canvas.create_text(self.app.width/2, self.app.margin,
                           text = title, font = titleFont)

class InfoMode(Mode):
    def appStarted(self):
        pass

    def keyPressed(self, event):
        #if event.key == "Escape" or event.key == "?" or event.key == "/":
        self.app.setActiveMode(self.app.configMode)

    def redrawAll(self, canvas):
        canvas.create_text(self.app.width // 2, self.app.margin,
                           text = "Build-Your-Own Multi-Layer-Perceptron",
                           font = "Helvetica 20")

class ConfigMode(Mode):
    def appStarted(self):
        self.warningMessages = dict()

    def keyPressed(self, event):
        newDims = None
        if event.key == "Right":
            if self.app.network.numLayers < 9:
                self.app.network.dims.append(1)
                newDims = self.app.network.dims
        elif event.key == "Left":
            if len(self.app.network.dims) > 1:
                self.app.network.dims.pop()
                newDims = self.app.network.dims
        elif event.key == "Up":
            if self.app.network.dims[-1] < 9:
                self.app.network.dims[-1] += 1
                newDims = self.app.network.dims
        elif event.key == "Down":
            if self.app.network.dims[-1] > 1:
                self.app.network.dims[-1] -= 1
                newDims = self.app.network.dims
        elif event.key == "Space":
            self.switchToTrainMode()
        elif event.key == "Tab":
            self.switchDataset()
        elif event.key == "a":
            self.switchActivationFunction()
        elif event.key == "/" or event.key == "?":
            self.app.setActiveMode(self.app.infoMode)
        elif event.key == "r":
            self.switchToDefaultParams()

        if newDims != None:
            self.app.network.resize(newDims)
            self.app.updateNetworkViewModel()

    def switchToDefaultParams(self):
        args = self.app.defaultParameters
        self.app.updateNetworkConfiguration(**args)

    def switchActivationFunction(self):
        self.app.activationFunctionIndex = (self.app.activationFunctionIndex + 1) % len(self.app.ACTIVATION_FUNCTIONS)

        args = {'activation' : self.app.activationFunctionIndex}
        self.app.updateNetworkConfiguration(**args)

    def switchDataset(self):
        self.app.datasetIndex = (self.app.datasetIndex + 1) % len(self.app.datasets)
        self.app.data = self.app.datasets[self.app.datasetIndex]

    def switchToTrainMode(self):
        inputsConfigured = self.app.network.dims[0] == self.app.data.numFeatures
        outputsConfigured = self.app.network.dims[-1] == self.app.data.numLabels

        if inputsConfigured:
            self.warningMessages['inputWarn'] = ""
        else:
            self.warningMessages['inputWarn'] = (
                (f'Dataset has {self.app.data.numFeatures} features but network' 
                +f' input layer has {self.app.network.dims[0]} nodes.'))

        if outputsConfigured:
            self.warningMessages['outputWarn'] = ""
        else:
            self.warningMessages['outputWarn'] = (
                (f'Dataset has {self.app.data.numLabels} labels but network' 
                +f' output layer has {self.app.network.dims[-1]} nodes.'))

        if self.app.network.numLayers > 2:
            self.warningMessages['layerWarn'] = ""
        else:
            self.warningMessages['layerWarn'] = (
                (f'Network has {self.app.network.numLayers} layers but'
                ' requires at least 3 for training.')
            )

        for message in self.warningMessages.values():
            if message != "":
                return
        self.app.setActiveMode(self.app.trainMode)

    def redrawAll(self, canvas):
        self.app.drawNetwork(canvas, visualizeParams = False, doStipple = False)
        s = ("Configuration Mode\n\n"
             "Press right and left arrow keys to add and remove layers.\n"
             "Press up and down arrow keys to add and remove neurons.\n"
             "Press a to change activation function.\n"
             "Press tab to change datasets.\n"
             "Press r for default settings.\n"
             "Press space to begin training.\n")

        for message in self.warningMessages.values():
            s += "\n" + message
        canvas.create_text(self.app.margin, self.app.margin,
                           text = s,
                           anchor = "nw")

        self.app.drawConfigInfo(canvas)

class TrainMode(Mode):
    LOSS_GRAPH_X_MIN = 20
    LOSS_GRAPH_COLOR = "Blue"
    LOSS_GRAPH_ROWS = 5
    LOSS_GRAPH_COLS = 5

    def appStarted(self):
        self.mouse = (0,0)
        self.timerDelay = 100
        # You may click on neurons to toggle on or off visualizing their outputs
        # You may hover over a neuron to see the specific values associated with
        # it
        self.isVisualizing = False
        self.doSoloHover = False
        self.hoveredNode = None
        self.toggleVisualization(forceOn = True)
        self.isTraining = False
        self.manualStep = 1
        self.autoStep = 50
        self.showHelp = True
        self.currentLoss = 0
        self.initializeLossGraph()

    def modeActivated(self):
        self.doSoloHover = False
        self.toggleVisualization(forceOn = True)

    def keyPressed(self, event):
        if event.key == "Right":
            self.doTraining(self.manualStep)
        elif event.key == "Space":
            self.isTraining = False if self.isTraining else True
        elif event.key == "r":
            self.restartTraining()
        elif event.key == "Up":
            self.app.alpha += 0.5
        elif event.key == "Down":
            if self.app.alpha >= 0.5:
                self.app.alpha -= 0.5
        elif event.key == "t":
            self.toggleHoveringMode()
            #self.toggleVisualization()
        elif event.key == "Escape":
            self.switchToConfigMode()
        elif event.key == "h":
            self.showHelp = False if self.showHelp else True
        elif event.key == "b":
            self.app.debug = False if self.app.debug else True
    
    def mousePressed(self, event):
        r = self.app.r
        for node in self.app.nodeCoordinatesSet:                
            if pointInCircle(r, node, (event.x, event.y)):
                if node in self.selectedNodeCoords and self.isVisualizing:
                    self.selectedNodeCoords.remove(node)
                else:
                    self.doSoloHover = False
                    self.selectedNodeCoords.add(node)

    def mouseMoved(self, event):
        r = self.app.r
        mouse = (event.x, event.y)
        self.mouse = mouse
        for node in self.app.nodeCoordinatesSet:
            if pointInCircle(r, node, mouse):
                self.hoveredNode = node
                if self.doSoloHover:
                    self.setSoloNode(node)
            elif (self.hoveredNode != None and not
                  pointInBounds(mouse, self.app.networkViewBounds)):
                self.hoveredNode = None
                self.toggleVisualization(forceOn = True)

    def setSoloNode(self, node):
        self.toggleVisualization(forceOff = True)
        self.selectedNodeCoords = set((node,))
    
    def toggleHoveringMode(self):
        if self.doSoloHover:
            self.toggleVisualization(forceOn = True)
            self.doSoloHover = False
        else:
            self.toggleVisualization(forceOff = True)
            self.doSoloHover = True
    
    def toggleVisualization(self, forceOff = False, forceOn = False):
        if forceOn:
            self.enableVisualization()
        elif forceOff:
            self.disableVisualization()
        else:
            self.isVisualizing = False if self.isVisualizing else True
            if self.isVisualizing:
                self.enableVisualization()
            else:
                self.disableVisualization()

    def enableVisualization(self):
        self.isVisualizing = True
        coords = self.app.nodeCoordinates
        self.selectedNodeCoords = set(flatten2dList(self.app.nodeCoordinates))

    def disableVisualization(self):
        self.isVisualizing = False
        self.selectedNodeCoords = set()

    def switchToConfigMode(self):
        self.restartTraining()
        self.hoveredNode = None
        self.toggleVisualization(forceOn = True)
        self.app.setActiveMode(self.app.configMode)
    
    def restartTraining(self):
        self.isTraining = False
        self.app.network.initializeParameters()
        self.initializeLossGraph()
    
    def initializeLossGraph(self):
        self.lossPerEpoch = []
        self.maxLoss = -1
        self.minLoss = 100
        self.calculateLoss()

    # Performs training
    def timerFired(self):
        if self.isTraining:
            self.doTraining(1)

    # Calculates the loss of the network on the test set
    def calculateLoss(self):
        cost = 0
        self.latestModelIO = []
        for example in self.app.data.test:
            x = example[0]
            yHat = self.app.network.forwardPropagation(x)
            y = example[1]
            self.latestModelIO.append((x, yHat))
            cost += self.app.network.cost(y, yHat)
        self.currentLoss = cost / len(self.app.data.train)
        epochLossTuple = (self.app.network.numTrainingIterations, self.currentLoss)
        self.lossPerEpoch.append(epochLossTuple)
        self.updateLossMaxMin()

    # Updates the maximum and minimum recorded loss for the current training
    # session. Must be called after every loss calculation as it only uses
    # the current loss for comparison.
    def updateLossMaxMin(self):
        if self.currentLoss > self.maxLoss:
            self.maxLoss = self.currentLoss
        elif self.currentLoss <= self.minLoss:
            self.minLoss = self.currentLoss

    # Performs the specified number of training iterations and calculates the loss
    # afterwards
    def doTraining(self, iterations):
        self.app.network.train(self.app.data.train, iterations, self.app.alpha)
        print(f'{self.app.network.test(self.app.data.test)}/{len(self.app.data.test)}')
        self.calculateLoss()

    # Maps a percentage of the color legend (0.00 - 1.00) to the color at that
    # relative point along the legend
    def mapPercentToLegendColor(self, percent):
        percent *= 2
        if 0 <= percent < 1:
            return self.app.rgbString(255, 50, int(percent*255))
        else:
            return self.app.rgbString(int((2 - percent)*255), 50, 255)

    # Draw axes and associated values for loss graph
    def drawLossGraphGrid(self, canvas, h, w, tY, bY, rX, lX):
        # Left Axis Title and end-point values

        # Learned how to get the function name as a string using __name__ here:
        # https://stackoverflow.com/questions/251464/how-to-get-a-function-name-as-a-string
        lossFunction = self.app.network.cost.__name__
        canvas.create_text(lX - 25, (bY - h/2), text = f'Loss ({lossFunction})',
                            angle = 90, anchor = "s")
        yMax = self.maxLoss
        canvas.create_text(lX, tY, text = '%0.2f' % self.maxLoss, anchor = "s")
        canvas.create_text(lX, bY, text = "0", anchor = "ne")

        # Intermediate values for left axis
        dRow = h / self.LOSS_GRAPH_ROWS
        for row in range(self.LOSS_GRAPH_ROWS - 1, 0, -1):
            canvas.create_line(lX, tY+row*dRow,
                               lX+w, tY+row*dRow,
                               fill = "grey")
            tickVal = "%0.2f" % ((1 - (row / self.LOSS_GRAPH_ROWS))*yMax)
            canvas.create_text(lX, tY+row*dRow, text = tickVal, anchor = 'e',
                               font = "Helvetica 8")

        # Bottom Axis Title and end-point values
        canvas.create_text(lX + w/2, bY, text = '\nIteration', anchor = "n")
        xMax = max(self.LOSS_GRAPH_X_MIN, self.app.network.numTrainingIterations)
        canvas.create_text(rX, bY, text = xMax,
                            anchor = "n")
        
        # Intermediate values for bottom axis
        dCol = w / self.LOSS_GRAPH_COLS
        tickLen = min(h, w) / 25
        for col in range(1, self.LOSS_GRAPH_COLS):
            canvas.create_line(col*dCol+lX, tY,
                               col*dCol+lX, bY,
                               fill = "grey")
            tickVal = "%0.0f" % (roundHalfUp((col / self.LOSS_GRAPH_COLS)*xMax))
            canvas.create_text(col*dCol+lX, bY, text = tickVal, anchor = "n")

    def drawHoverTooltip(self, canvas):
        w = self.app.height // 4
        tY = self.app.margin + self.app.height // 3 # just below loss graph
        lX = self.app.width - self.app.margin*2  
        
        if self.hoveredNode == None:
            s = "Hover over node to view parameters."
            lX = self.app.width - self.app.height // 4 - self.app.margin
            canvas.create_text(lX, tY, text = s, anchor = "nw")
        else:
            x, y = self.hoveredNode
            myNodeIndex = self.findNodeIndexFromCoordinates(x, y)
            if myNodeIndex == None:
                s = "Can't find node."
            else:
                s = self.readParametersAtNodeIndex(myNodeIndex)
            canvas.create_text(lX, tY, text = s, anchor = "ne")

    def readParametersAtNodeIndex(self, nodeIndex):
        s = weightString = biasString = labelString = ""
        layer, node = nodeIndex
        if layer == 0:
            s += "Input layer node.\n\n"
            weightString = self.readWeightsAtNodeIndex(nodeIndex)
        elif layer == self.app.network.numLayers - 1:
            s += "Output layer node.\n\n"
            labelString = self.readLabelAtOutputNode(node)
            biasString = self.readBiasesAtNodeIndex(nodeIndex)
        else:
            s += "Hidden layer node.\n\n"
            weightString = self.readWeightsAtNodeIndex(nodeIndex)
            biasString = self.readBiasesAtNodeIndex(nodeIndex)
        
        return s + labelString + weightString + biasString
    
    def readLabelAtOutputNode(self, node):
        return f'Label: {self.app.data.labels[node]}\n\n'
    
    def readWeightsAtNodeIndex(self, nodeIndex):
        s = ""
        layer, node = nodeIndex
        outgoingWeights = getColumn(self.app.network.w[layer], node)
        for outgoingWeightIndex in range(len(outgoingWeights)):
            weightVal = outgoingWeights[outgoingWeightIndex]
            truncatedWeightValString = '%0.4f' % weightVal
            s += f"w{outgoingWeightIndex} = {truncatedWeightValString}\n"
        return s + '\n'
    
    def readBiasesAtNodeIndex(self, nodeIndex):
        s = ""
        layer, node = nodeIndex
        # No bias term in first layer
        for bias in self.app.network.b[layer - 1][node]:
            biasVal = bias
            truncatedBiasValString = '%0.4f' % biasVal
            s += f"b = {truncatedBiasValString}"
        return s + '\n'
        
    def findNodeIndexFromCoordinates(self, x, y):
        for layerIndex in range(len(self.app.nodeCoordinates)):
            for nodeIndex in range(len(self.app.nodeCoordinates[layerIndex])):
                nodeCoord = self.app.nodeCoordinates[layerIndex][nodeIndex]
                if nodeCoord == (x, y):
                    return (layerIndex, nodeIndex)
        

    # Draws loss graph in top left
    def drawLossGraph(self, canvas):
        h = w = self.app.height // 4            # height, width
        tY = self.app.margin                    # top Y
        bY = self.app.margin + h                # bottom Y
        rX = self.app.width - self.app.margin   # right X
        lX = rX - w                             # left X

        canvas.create_rectangle(lX, bY, rX, tY)
        self.drawLossGraphGrid(canvas, h, w, tY, bY, rX, lX)

        iteration = max(self.LOSS_GRAPH_X_MIN, self.app.network.numTrainingIterations)

        for i in range(len(self.lossPerEpoch) - 1):
            x1, y1 = self.lossPerEpoch[i]
            x2, y2 = self.lossPerEpoch[i + 1]
            
            x1Scaled = (x1 / iteration)*w + lX
            y1Scaled = (1 - y1 / self.maxLoss)*w + tY

            x2Scaled = (x2 / iteration)*w + lX
            y2Scaled = (1 - y2 / self.maxLoss)*w + tY

            canvas.create_line(x1Scaled, y1Scaled, x2Scaled, y2Scaled,
                               fill = self.LOSS_GRAPH_COLOR)

    # Draws a color gradient with TkInter lines of changing color tone
    def drawColorLegend(self, canvas):
        legendHeight = self.app.height // 4
        legendWidth = self.app.width // 30
        legendTopY = self.app.height - self.app.margin - legendHeight
        legendBottomY = self.app.height - self.app.margin
        legendRightX = self.app.margin + legendWidth
        legendLeftX = self.app.margin
        canvas.create_rectangle(legendLeftX, legendTopY,
                                legendRightX, legendBottomY)
        numPixels = legendBottomY - legendTopY
        for px in range(numPixels):
            percent = px / numPixels
            canvas.create_line(legendLeftX, legendTopY + px, legendRightX, legendTopY + px,
                                fill = self.mapPercentToLegendColor(percent))
        
        canvas.create_text(legendRightX, legendTopY, text = " +", anchor = "w")
        canvas.create_text(legendRightX, legendTopY + numPixels/2, text = " 0", anchor = "w")
        canvas.create_text(legendRightX, legendBottomY, text = " -", anchor = "w")

    def redrawAll(self, canvas):
        self.app.drawNetwork(canvas)
        canvas.create_text(self.app.width // 2, 50,
                           text = f'Iteration: {self.app.network.numTrainingIterations}')

        lossString = "%.7f" % self.currentLoss
        s = (f'Training Mode\n\n'
            +f'Learning rate: {self.app.alpha}\n'
            +f'Loss on validation set: {lossString}\n\n')

        if self.showHelp:
            s += ('Press h to hide help.\n\n'
                +'Press space to start or pause training.\n'
                +f'Press the right arrow key to skip forward {self.manualStep} iterations\n'
                +'Press r to reset weights and biases.\n'
                +'Press up or down to increase or decrease the learning rate.\n'
                +'Press escape to go back to configuration mode.\n')
            self.drawColorLegend(canvas)
        else:
            s += 'Press h to show help.\n\n'

        canvas.create_text(self.app.margin, self.app.margin,
                           text = s,
                           anchor = "nw")

        if self.isTraining:
            text = "Training..."
        else:
            text = "Training paused."

        canvas.create_text(self.app.width // 2, self.app.height - self.app.margin,
                            text = text)
        self.drawLossGraph(canvas)
        self.drawHoverTooltip(canvas)
        self.app.drawConfigInfo(canvas)
        canvas.create_oval(self.mouse[0]- 5, self.mouse[1] - 5,
                           self.mouse[0]+ 5, self.mouse[1] + 5)

class NeuralNetworkApp(ModalApp):
    ACTIVATION_FUNCTION_NAMES = ["Logistic", "TanH"]
    ACTIVATION_FUNCTIONS = {"Logistic" : logistic, "TanH" : tanH}
    NODE_RADIUS_RATIO = 1/40
    # Copied from:
    # https://www.cs.cmu.edu/~112/notes/notes-graphics-part2.html#customColors
    @staticmethod
    def rgbString(red, green, blue):
        return "#%02x%02x%02x" % (red, green, blue)

    # Starts the Neural Network App
    def appStarted(self):
        self.debug = False
        self.margin = 50
        self.datasetIndex = 0
        self.activationFunctionIndex = 0
        self.datasets = self.findAllDatasetsInDirectory()
        self.data = self.datasets[0]
        self.alpha = 1
        self.defaultParameters = {'dims' : (4, 5, 5, 2),
                                  'activation' : self.activationFunctionIndex,
                                  'dataset' : self.datasetIndex}
        self.network = NeuralNetwork([4, 5, 5, 2], logistic)
        self.updateNetworkViewModel()
        self.configMode = ConfigMode()
        self.trainMode = TrainMode()
        self.infoMode = InfoMode()
        self.setActiveMode(self.configMode)
    
    # Takes variable arguments, updates dimensions, activation, and/or dataset
    def updateNetworkConfiguration(self, **kwargs):
        dims = list(kwargs.get('dims', self.network.dims))
        newActivationIndex = kwargs.get('activation',
                                        self.activationFunctionIndex)
        self.activationFunctionIndex = newActivationIndex
        activationFuncName = self.ACTIVATION_FUNCTION_NAMES[newActivationIndex]
        activation = self.ACTIVATION_FUNCTIONS[activationFuncName]
        self.network = NeuralNetwork(dims, activation)
        self.datasetIndex = kwargs.get('dataset', self.datasetIndex)

    # Finds all CSV files in the working directory
    def findAllDatasetsInDirectory(self):
        filepaths = listFiles('datasets', suffix = '.csv')
        datasetList = []
        for path in filepaths:
            newDataset = Dataset(path)
            datasetList.append(newDataset)
        return datasetList

    # Converts parameter to a color and returns RGB value
    def weightToColor(self, weight, biasTerm = False):
        scaledWeight = int(math.e**(abs(weight)+2)+1)
        if biasTerm:
            scaledWeight = int(math.e**(abs(weight)+2)+1)
        weightMappedToChannel = (255 - getInBounds(scaledWeight, 0, 255))
        if weight < 0:
            return NeuralNetworkApp.rgbString(weightMappedToChannel, 50, 255)
        return NeuralNetworkApp.rgbString(255, 50, weightMappedToChannel)

    # Draws current dataset and activation function name
    def drawConfigInfo(self, canvas):
        
        s = f'Dataset: {self.data}\n'
        activationFuncName = self.ACTIVATION_FUNCTION_NAMES[self.activationFunctionIndex]
        s += f'Activation Function: {activationFuncName}'
        canvas.create_text(self.width - self.margin, self.height - self.margin,
                            anchor = "se",
                            text = s)

    # Refresh network view model on window resize with updateNetworkViewModel()
    def sizeChanged(self):
        self.updateNetworkViewModel()
    
    # Recalculate node radius and network view coordinates
    def updateNetworkViewModel(self):
        self.r = min(self.width*self.NODE_RADIUS_RATIO,
                     self.height*self.NODE_RADIUS_RATIO)
        self.regenerateNodeCoordinates()
        self.regenerateNetworkViewBounds()
        self.nodeCoordinatesSet = set(flatten2dList(self.nodeCoordinates))

    # Draws the bias for a specified layer and node
    def drawBias(self, canvas, l, n, coords, r, visualizeParams, visualizeMe):
        # First layer has no bias term.
        if l == 0 or not visualizeMe:
            biMagnitude = 1
            if not visualizeParams or not visualizeMe or l != 0:
                bColor = 'white'
            else:
                bColor = 'lightgray'
        else:
            bi = self.network.b[l-1][n][0]
            biMagnitude = abs(bi)
            bColor = self.weightToColor(bi, biasTerm = True)
        cx, cy = coords[l][n]
        canvas.create_oval(cx-r, cy-r, cx+r, cy+r, width = biMagnitude,
                           fill = bColor)
        
    # Draws the weights for a specified layer and node
    # visualizeMe takes precedence over visualizeParams
    def drawWeights(self, canvas, l, n, coords, r, visualizeParams, visualizeMe, doStipple):
        cx, cy = coords[l][n]
        if l == len(self.network.dims) - 1:
            return
        for n2 in range(len(coords[l+1])):
            if visualizeMe:
                wij = self.network.w[l][n2][n]
                wColor = self.weightToColor(wij)
                wijMagnitude = abs(wij)
                stipple = ''
            else:
                wColor = None
                wijMagnitude = 1
                stipple = 'gray75' if doStipple else ''
            
            cx2, cy2 = coords[l+1][n2]
            canvas.create_line(cx+r, cy, cx2-r, cy2, width = wijMagnitude,
                                fill = wColor, stipple = stipple)
    
    # Draws the network onto the canvas, parameter visualization optional
    def drawNetwork(self, canvas, visualizeParams = True, doStipple = True):
        r = self.r  # Node radius
        coords = self.nodeCoordinates
        for l in range(len(coords)):
            for n in range(len(coords[l])):
                if visualizeParams:
                    xy = coords[l][n]
                    visualizeMe = xy in self.trainMode.selectedNodeCoords
                else:
                    visualizeMe = False
                self.drawBias(canvas, l, n, coords, r,
                                visualizeParams, visualizeMe)

                cx, cy = coords[l][n]
                self.drawWeights(canvas, l, n, coords, r, visualizeParams,
                                 visualizeMe, doStipple)

    def regenerateNodeCoordinates(self):
        nodes = []
        cy = self.height / 2                        # Network center y
        cx = self.width / 2                         # Network center x
        cl = (len(self.network.dims) - 1) / 2       # Center layer index
        dl = self.width / 9                         # Layer x spacing
        dn = self.height / 9                        # Node y spacing
        for l in range(len(self.network.dims)):
            nodes.append([])
            lc = l - cl                             # Layer center x
            cn = (self.network.dims[l] - 1) / 2
            for n in range(self.network.dims[l]):
                nc = n - cn                         # Node center y
                nodeCoord = (cx + lc*dl, cy+ nc*dn)
                nodes[-1].append(nodeCoord)
        self.nodeCoordinates = nodes
    
    def regenerateNetworkViewBounds(self):
        coords = self.nodeCoordinates
        # Nested list comprehension
        x = [point[0] for layer in coords for point in layer]
        y = [point[1] for layer in coords for point in layer]
        ax1, ax2 = min(x) - self.r, max(x) + self.r
        ay1, ay2 = min(y) - self.r, max(y) + self.r
        self.networkViewBounds = (ax1, ay1, ax2, ay2)

if __name__ == "__main__":
    NeuralNetworkApp(width = 1700, height = 900)