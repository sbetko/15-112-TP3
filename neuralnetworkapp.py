# pylint: disable=no-member

# This file houses the program driver, the main app class
# NeuralNetworkApp extending ModalApp, as well as all of its modes,
# Start Mode, Configuration Mode, Training Mode, and Testing Mode.

import math, random, copy, numbers, os

# Taken from course site: https://www.cs.cmu.edu/~112/notes/cmu_112_graphics.py
from cmu_112_graphics import Mode, ModalApp
from tkinter import *

# Helper functions adapted from course website, citations included inside
from helpers112 import *

# My imports and code
from mydatasetlib import Dataset
from myneuralnetwork import NeuralNetwork
from mymathlib import *
from mybuttonlib import *
from mygraphicslib import *

# Taken from here https://docs.python.org/3/library/pickle.html
# For serialization/deserialization of the network
import pickle

# Things To Do:
# DONE: Add CSV read-support
# DONE: Add a graph for the loss function with matplotlib
# DONE: Add mouse-hover to view parameter
# DONE: Add validation set
# DONE: Add a test mode
# DONE: Add model import / export
# DONE: Add a mouse-based GUI with TkInter
# TODO: Implement Stochastic Gradient Descent
#       - With variable batch size too
# TODO: Add a regularization parameter
# TODO: Add a training set generator with choice of mathematical function on a
#       chosen domain (with respective test set generator)
# TODO: Add support for regression
# TODO: Speed up the matrix math a little bit

# The first mode the user sees
class StartMode(Mode):
    BACKGROUND_FREQ = 900
    BACKGROUND_RGB1 = "255050000"
    BACKGROUND_RGB2 = "000050255"
    def appStarted(self):
        self.app.initDisplayPieceNetwork()
        self.panels = []
        self.configureMainPanel()
        self.app.allPanels.append(self.panels)
        self.timer = 0
    
    # Initializes the background
    def modeActivated(self):
        self.app.initDisplayPieceNetwork()
    
    # Makes a lightshow on the splash screen
    def timerFired(self):
        self.timer += self.app.timerDelay
        if self.timer % self.BACKGROUND_FREQ == 0:
            self.app.network.initializeParameters()

    # Checks if user interacted with any panels
    def mousePressed(self, event):
        point = (event.x, event.y)
        self.app.doPanelPress(point, self.panels)

    # Initializes the main center panel
    def configureMainPanel(self):
        cx, cy = self.app.width/2, self.app.height/2
        width, height = self.app.width/5, self.app.height/4
        self.mainPanel = Panel(cx, cy, width, height)
        
        configModeButton = Button(self.switchToConfigMode, "Design Network")
        importModelButton = Button(self.importModel, "Import Model")

        self.mainPanel.addButton(configModeButton)
        self.mainPanel.addButton(importModelButton)
        self.panels.append(self.mainPanel)

    # Switches to the network design mode
    def switchToConfigMode(self):
        self.app.initNetwork()
        self.app.setActiveMode(self.app.configMode)
    
    # Follows Python 3 docs here for unpickling objects:
    # https://docs.python.org/3/library/pickle.html#pickling-class-instances
    def importModel(self):
        fileName = self.app.getUserInput(
            "Please enter the filename with no extension\n"
            +"(Must be in program's working directory)"
            )
        myNetwork = None
        try:
            with open(fileName + ".nn", 'rb') as f :
                myNetwork = pickle.load(f)
            
            self.loadModel(myNetwork)
        except:
            self.app.showMessage(
                "Failed to load from file, please check that"
                " you entered the correct filename."
                )
    
    def loadModel(self, myNetwork):
        self.app.network = myNetwork
        self.app.updateNetworkViewModel()
        self.app.setActiveMode(self.app.trainMode)
    
    def redrawAll(self, canvas):
        title = "Build Your Own Neural Network"
        titleFont = "Helvetica 26"
        canvas.create_text(self.app.width/2, self.app.height / 10,
                           text = title, font = titleFont)
        self.app.drawNetwork(canvas, visualizeParams = True,
                             rgb1 = self.BACKGROUND_RGB1,
                             rgb2 = self.BACKGROUND_RGB2)
        self.mainPanel.drawPanelVertical(canvas)

class ConfigMode(Mode):
    def appStarted(self):
        self.configurePanels()
        self.warningMessages = dict()
    
    # Configures the main panels 
    def configurePanels(self):
        self.panels = []
        self.app.configureBackPanel("<- Main Menu", self)
        self.configureControlPanel()
        self.app.configureNextPanel("Train ->", self)
        self.app.allPanels.append(self.panels)

    # Initializes the control panel with dataset and activation function
    # togglers, and a configuration reset button
    def configureControlPanel(self):
        width = self.app.width / 5
        height = self.app.height / 10
        self.controlPanel = Panel(self.app.margin,
                                  self.app.height - self.app.margin,
                                  width, height, anchor = "sw")
        resetTitle = "Default configuration"
        self.resetButton = Button(self.switchToDefaultParams, resetTitle)
        datasetTitle = "Change dataset"
        self.datasetButton = Button(self.switchDataset, datasetTitle)
        activationTitle = "Change activation function"
        self.activationButton = Button(self.switchActivationFunction,
                                         activationTitle)
        self.controlPanel.addButton(self.datasetButton)
        self.controlPanel.addButton(self.activationButton)
        self.controlPanel.addButton(self.resetButton)
        self.panels.append(self.controlPanel)
    
    # Sets app mode to the start mode
    def goBack(self):
        self.app.setActiveMode(self.app.startMode)
    
    # Sets app mode to the train mode
    def goNext(self):
        self.switchToTrainMode()
    
    # Checks if user clicked on any panels
    def mousePressed(self, event):
        point = (event.x, event.y)
        self.app.doPanelPress(point, self.panels)

    # Handles keyboard input and shortcuts
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
        elif event.key == "r":
            self.switchToDefaultParams()
        elif event.key == "Escape":
            self.goBack()
        if newDims != None:
            self.app.network.resize(newDims)
            self.app.updateNetworkViewModel()

    # Switches back to the default network parameters
    def switchToDefaultParams(self):
        args = self.app.defaultParameters
        self.app.updateNetworkConfiguration(**args)

    # Toggles through the activation functions
    def switchActivationFunction(self):
        self.app.activationFunctionIndex = (self.app.activationFunctionIndex + 1) % len(self.app.ACTIVATION_FUNCTIONS)

        args = {'activation' : self.app.activationFunctionIndex}
        self.app.updateNetworkConfiguration(**args)

    # Toggles through the active dataset
    def switchDataset(self):
        self.app.datasetIndex = (self.app.datasetIndex + 1) % len(self.app.datasets)
        self.app.data = self.app.datasets[self.app.datasetIndex]

    # Switches app mode to training mode if network architecture
    # is in agreement with the chosen dataset, otherwise configures the
    # appropriate warning messages
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

    # Draws onto the configuration mode canvas
    def redrawAll(self, canvas):
        self.app.drawNetwork(canvas, visualizeParams = False, doStipple = False)
        s = ("Configuration Mode\n\n"
             "Press right and left arrow keys to add and remove layers.\n"
             "Press up and down arrow keys to add and remove neurons.\n\n"
             "Press a to change activation function.\n"
             "Press tab to change datasets.\n"
             "Press r for default settings.\n"
             "Press space to begin training.\n")

        for message in self.warningMessages.values():
            s += "\n" + message
        canvas.create_text(self.app.margin, self.app.margin*2,
                           text = s,
                           anchor = "nw")

        self.app.drawConfigInfo(canvas)
        self.app.drawPanels(canvas, self.panels)

# Mode responsible for training the network
class TrainMode(Mode):
    # This is the minimum scale of the x-axis, if there are no points
    # with an x-value greater than this value then the graph will not
    # automatically resize to be smaller.
    LOSS_GRAPH_X_MIN = 20
    # This is the color of the curve drawn on the graph
    LOSS_GRAPH_COLOR = "Blue"
    LOSS_GRAPH_ROWS = 5
    LOSS_GRAPH_COLS = 5
    # Each tuple (RGB1, RGB2) specifies a continuous color gradient from
    # RGB1 to RGB2, defining the network's color visualization scheme
    COLOR_SCHEME_PRESETS = [("255050000", "000050255"),
                            ("255050050", "000255050"),
                            ("050255000", "050000255")]

    def appStarted(self):
        self.configurePanels()
        self.timerDelay = 100
        self.colorSchemeIndex = 0
        # You may click on neurons to toggle on or off visualizing their outputs
        # You may hover over a neuron to see the specific values associated with
        # it
        self.isVisualizing = False
        self.doSoloHover = False
        self.hoveredNode = None
        self.toggleVisualization(forceOn = True)
        self.isTraining = False
        self.manualStep = 1
        self.autoStep = 1
        self.showHelp = False
        self.currentLoss = 0
        # If the network had an export state then load in the state of its
        # training
        if self.app.network.exportState != None:
            self.loadTrainState()
        # Otherwise initialize anything that still needs to be initialized
        else:
            self.initializeLossGraph()

    # Configures all of the UI panels 
    def configurePanels(self):
        self.panels = []
        self.app.configureBackPanel("<- Configuration", self)
        self.configureControlPanel()
        self.configureStartPanel()
        self.app.configureNextPanel("Test ->", self)
        self.app.allPanels.append(self.panels)
    
    # Configures the control panel that holds the learning rate button,
    # reset training button, hover visualization mode toggle button, and
    # color scheme change button.
    def configureControlPanel(self):
        width = self.app.width / 7
        height = self.app.height / 5
        self.controlPanel = Panel(self.app.margin, self.app.height/2.2, width, height,
                             anchor = "nw")
        self.learningRateButton = Button(self.setLearningRate, "Set learning rate")
        self.resetButton = Button(self.reset, "Reset")
        hoverVizTitle = "Change visualization mode\n(Hint: click the neurons)"
        self.hoverVizButton = Button(self.toggleHoveringMode, hoverVizTitle)
        self.colorButton = Button(self.changeColorScheme, "Change color scheme")
        self.controlPanel.addButton(self.learningRateButton)
        self.controlPanel.addButton(self.resetButton)
        self.controlPanel.addButton(self.hoverVizButton)
        self.controlPanel.addButton(self.colorButton)
        self.panels.append(self.controlPanel)
    
    # Configure the panel that holds the start/stop training toggle button
    def configureStartPanel(self):
        width = self.app.width / 10
        height = self.app.height / 20
        self.startPanel = Panel(self.app.width/2, self.app.height - self.app.margin,
                           width, height)
        self.startPanel.backgroundColor = "lightgray"
        title = "Start Training"
        self.startStopButton = Button(self.toggleTraining, title, )
        self.startStopButton.isToggleButton = True
        self.startStopButton.activeText = "Pause Training"
        self.startPanel.addButton(self.startStopButton)
        self.panels.append(self.startPanel)

    # Go to the previous page
    def goBack(self):
        self.switchToConfigMode()
    
    # Go to the next page
    def goNext(self):
        self.isTraining = False
        self.app.setActiveMode(self.app.testMode)
    
    # Specifies the learning rate
    def setLearningRate(self):
        s = "Enter a number between 0 and 10"
        userInput = self.app.getUserInput(s)
        if userInput == None:
            return
        try:
            learningRate = getInBounds(float(userInput), 0, 10)
            self.app.alpha = learningRate
        except:
            self.app.showMessage("Not a valid number! Please try again.")

    # Resets the training mode
    def reset(self):
        self.colorSchemeIndex = 0
        self.restartTraining()

    # Toggles through the colors scheme presets
    def changeColorScheme(self):
        self.colorSchemeIndex = ((self.colorSchemeIndex + 1)
                                 % len(self.COLOR_SCHEME_PRESETS))
    
    # Toggles training on or off
    def toggleTraining(self):
        self.isTraining = False if self.isTraining else True

    # Loads in the training state of the network
    def loadTrainState(self):
        state = self.app.network.exportState
        self.app.data = state['data']
        self.lossPerEpoch = state['lossPerEpoch']
        self.currentAccuracy = state['currentAccuracy']
        self.currentLoss = state['currentLoss']
        self.maxLoss = state['maxLoss']

    # Called when the mode is activated
    def modeActivated(self):
        self.doSoloHover = False
        self.toggleVisualization(forceOn = True)

    # Checks for keyboard input and shortcuts
    def keyPressed(self, event):
        if event.key == "Right":
            self.doTraining(self.manualStep)
        elif event.key == "Space":
            self.startStopButton.activate()
        elif event.key == "r":
            self.restartTraining()
        elif event.key == "Up":
            if self.app.alpha < 10:
                self.app.alpha += 0.5
        elif event.key == "Down":
            if self.app.alpha >= 0.5:
                self.app.alpha -= 0.5
        elif event.key == "t":
            self.toggleHoveringMode()
        elif event.key == "Escape":
            self.backButton.activate()
        elif event.key == "h":
            self.showHelp = False if self.showHelp else True
        elif event.key == "Enter":
            self.nextButton.activate()
    
    # Checks for panel or node presses
    def mousePressed(self, event):
        r = self.app.r
        point = (event.x, event.y)
        if self.app.doPanelPress(point, self.panels):
            return

        for node in self.app.nodeCoordinatesSet:                
            if pointInCircle(r, node, (event.x, event.y)):
                if node in self.selectedNodeCoords and self.isVisualizing:
                    self.selectedNodeCoords.remove(node)
                else:
                    self.doSoloHover = False
                    self.selectedNodeCoords.add(node)

    # Checks for node hovering and sets the hovered node
    # appropriately
    def mouseMoved(self, event):
        r = self.app.r
        mouse = (event.x, event.y)
        for node in self.app.nodeCoordinatesSet:
            if pointInCircle(r, node, mouse):
                self.hoveredNode = node
                if self.doSoloHover:
                    self.setSoloNode(node)
            elif (self.hoveredNode != None and not
                  pointInBounds(mouse, self.app.networkViewBounds)):
                self.hoveredNode = None
                self.toggleVisualization(forceOn = True)

    # Sets the node to be solo-viewed
    def setSoloNode(self, node):
        self.toggleVisualization(forceOff = True)
        self.selectedNodeCoords = set((node,))
    
    # Toggles the hover-on-neuron-to-view-solo mode
    def toggleHoveringMode(self):
        if self.doSoloHover:
            self.toggleVisualization(forceOn = True)
            self.doSoloHover = False
        else:
            self.toggleVisualization(forceOff = True)
            self.doSoloHover = True
    
    # Toggles visualization with option to force specific state
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

    # Turns visualization back on
    def enableVisualization(self):
        self.isVisualizing = True
        coords = self.app.nodeCoordinates
        self.selectedNodeCoords = set(flatten2dList(self.app.nodeCoordinates))

    # Disables visualizing the parameters
    def disableVisualization(self):
        self.isVisualizing = False
        self.selectedNodeCoords = set()

    # Goes back to configuration mode
    def switchToConfigMode(self):
        self.restartTraining()
        self.hoveredNode = None
        self.toggleVisualization(forceOn = True)
        self.app.setActiveMode(self.app.configMode)
    
    # Restarts the training process
    def restartTraining(self):
        self.isTraining = False
        self.app.alpha = 1
        self.app.network.initializeParameters()
        self.initializeLossGraph()
    
    # Initializes the graph of the loss function vs iteration
    def initializeLossGraph(self):
        self.lossPerEpoch = []
        self.maxLoss = -1
        self.testNetworkAgainstValidationSet()

    # Performs training
    def timerFired(self):
        if self.isTraining:
            self.doTraining(self.autoStep)

    # Calculates the loss and accuracy of the network on the validation set
    def testNetworkAgainstValidationSet(self):
        cost = 0
        numCorrect = 0

        for example in self.app.data.validation:
            # Loss calculation
            x = example[0]
            predictedY = self.app.network.forwardPropagation(x)
            y = example[1]
            cost += self.app.network.cost(y, predictedY)

            # Accuracy calculation
            highestPercentage = -1
            winningLabelIndex = None
            for i in range(len(predictedY)):
                if predictedY[i][0] > highestPercentage:
                    highestPercentage = predictedY[i][0]
                    winningLabelIndex = i

            # test against true label
            if y[winningLabelIndex][0] == [1]:
                numCorrect += 1

        self.currentAccuracy = numCorrect / len(self.app.data.validation)
        self.currentLoss = cost / len(self.app.data.validation)
        epochLossTuple = (self.app.network.numTrainingIterations,
                          self.currentLoss)
        self.lossPerEpoch.append(epochLossTuple)
        self.updateLossMax()

    # Updates the maximum recorded loss for the current training
    # session. Must be called after every loss calculation as it only uses
    # the current loss for comparison.
    def updateLossMax(self):
        if self.currentLoss > self.maxLoss:
            self.maxLoss = self.currentLoss

    # Performs the specified number of training iterations and calculates the
    # loss afterwards
    def doTraining(self, iterations):
        self.app.network.train(self.app.data.train, iterations, self.app.alpha)
        self.testNetworkAgainstValidationSet()

    # Draw axes and associated values for loss graph
    def drawLossGraphGrid(self, canvas, h, w, tY, bY, rX, lX):
        # Left Axis Title and end-point values

        # Learned how to get the function name as a string using __name__ here:
        # https://docs.python.org/3/library/stdtypes.html#special-attributes
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

    # Draws the values associated with the currently hovered on node onto the 
    # canvas
    def drawHoverTooltip(self, canvas):
        w = self.app.height // 4
        tY = self.app.margin + self.app.height // 2 # just below loss graph
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

    # Reads the parameters values at the node index in the network
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
    
    # Reads the name of the label at the output node
    def readLabelAtOutputNode(self, node):
        return f'Label: {self.app.data.labels[node]}\n\n'
    
    # Reads the weights at a node index in the network
    def readWeightsAtNodeIndex(self, nodeIndex):
        s = ""
        layer, node = nodeIndex
        outgoingWeights = getColumn(self.app.network.w[layer], node)
        for outgoingWeightIndex in range(len(outgoingWeights)):
            weightVal = outgoingWeights[outgoingWeightIndex]
            truncatedWeightValString = '%0.4f' % weightVal
            s += f"w{outgoingWeightIndex} = {truncatedWeightValString}\n"
        return s + '\n'
    
    # Reads the biaes at a node index in the network
    def readBiasesAtNodeIndex(self, nodeIndex):
        s = ""
        layer, node = nodeIndex
        # No bias term in first layer
        for bias in self.app.network.b[layer - 1][node]:
            biasVal = bias
            truncatedBiasValString = '%0.4f' % biasVal
            s += f"b = {truncatedBiasValString}"
        return s + '\n'
        
    # Finds the index of the node in the network given the view coordinates
    def findNodeIndexFromCoordinates(self, x, y):
        for layerIndex in range(len(self.app.nodeCoordinates)):
            for nodeIndex in range(len(self.app.nodeCoordinates[layerIndex])):
                nodeCoord = self.app.nodeCoordinates[layerIndex][nodeIndex]
                if nodeCoord == (x, y):
                    return (layerIndex, nodeIndex)

    # Draws loss graph in top left
    def drawLossGraph(self, canvas):
        h = w = self.app.height // 4            # height, width
        tY = self.app.margin * 3                # top Y
        bY = self.app.margin * 3 + h            # bottom Y
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
        rgb1, rgb2 = self.COLOR_SCHEME_PRESETS[self.colorSchemeIndex]
        drawColorGradientVertical(canvas, legendLeftX, legendTopY,
                                  legendWidth, legendHeight, rgb1, rgb2)
        top = " " + str(self.app.COLORIZATION_BOUND) + " (+)"
        bot = " " + str(-self.app.COLORIZATION_BOUND) + " (-)"
        canvas.create_text(legendRightX, legendTopY, text = top, anchor = "w")
        canvas.create_text(legendRightX, legendTopY + legendHeight/2, text = " 0", anchor = "w")
        canvas.create_text(legendRightX, legendBottomY, text = bot, anchor = "w")

    # Draws train mode animation onto the canvas
    def redrawAll(self, canvas):
        rgb1, rgb2 = self.COLOR_SCHEME_PRESETS[self.colorSchemeIndex]
        self.app.drawNetwork(canvas, rgb1 = rgb1, rgb2 = rgb2)
        canvas.create_text(self.app.width // 2, 50,
                           text = f'Iteration: {self.app.network.numTrainingIterations}')

        lossString = "%.7f" % self.currentLoss
        accuracy = self.currentAccuracy * 100
        accuracyString = "%.2f" % self.currentAccuracy
        s = (f'Training Mode\n\n'
            +f'Learning rate: {self.app.alpha}\n'
            +f'Loss on validation set: {lossString}\n'
            +f'Accuracy on validation set: {accuracyString}\n\n')

        if self.showHelp:
            s += ('Press h to hide keyboard shortcuts.\n\n'
                +'Press space to start or pause training.\n'
                +f'Press the right arrow key to skip forward {self.manualStep} iterations\n'
                +'Press r to reset weights and biases.\n'
                +'Press t to change visualization mode.\n'
                +'Press up or down to increase or decrease the learning rate.\n'
                +'Press enter to test the model.\n'
                +'Press escape to go back to configuration mode.\n')
        else:
            s += 'Press h to show keyboard shortcuts.\n\n'

        canvas.create_text(self.app.margin, self.app.margin*2,
                           text = s,
                           anchor = "nw")

        self.drawLossGraph(canvas)
        self.drawHoverTooltip(canvas)
        self.app.drawConfigInfo(canvas)
        self.drawColorLegend(canvas)
        self.app.drawPanels(canvas, self.panels)

# Mode for performance evaluation of the network and model export
class TestMode(Mode):
    # The RGB1 and RGB2 values for the confusion matrix color legend
    RGB1 = "247251255"
    RGB2 = "008048107"
    def appStarted(self):
        self.testData = self.app.data.test
        self.numExamples = len(self.testData)
        self.numLabels = self.app.data.numLabels
        self.maxMarginalCount = 0
        self.generateConfusionMatrix()
        self.precision = self.calculatePrecision()
        self.recall = self.calculateRecall()
        self.f1Score = self.calculateF1()
        self.precisionFormatted = '%0.5f' % self.precision
        self.recallFormatted = '%0.5f' % self.recall 
        self.f1ScoreFormatted = '%0.5f' % self.f1Score
        self.configurePanels()
    
    # Configures all of the panels
    def configurePanels(self):
        self.panels = []
        self.configureMainPanel()
        self.app.configureBackPanel("<- Main Menu", self)
        self.app.allPanels.append(self.panels)
    
    # Configures the main panel which holds the maine menu button
    # and export model button
    def configureMainPanel(self):
        x, y = self.app.width / 2, self.app.height / 2
        width = self.app.width / 15
        height = self.app.height / 20
        self.mainPanel = Panel(x, y, width, height, anchor = "nw")
        self.mainPanel.backgroundColor = "lightgray"
        exportModelButton = Button(self.exportModel, "Export Model")
        self.mainPanel.addButton(exportModelButton)
        self.panels.append(self.mainPanel)
    
    # Goes back to the start mode
    def goBack(self):
        self.app.setActiveMode(self.app.startMode)
    
    # I used the method described in the Python 3 docs to serialize an
    # object using the standard python pickle module
    # https://docs.python.org/3/library/pickle.html#module-pickle
    def exportModel(self):
        fileName = self.app.getUserInput("Please enter a filename") + '.nn'
        f = open(fileName, 'wb')
        self.app.network.exportState = {
            'data' : self.app.data,
            'lossPerEpoch' : self.app.trainMode.lossPerEpoch,
            'currentAccuracy' : self.app.trainMode.currentAccuracy,
            'currentLoss' : self.app.trainMode.currentLoss,
            'maxLoss' : self.app.trainMode.maxLoss}

        self.app.network.data = self.app.data
        self.app.network.lossPerEpoch = self.app.trainMode.lossPerEpoch
        pickle.dump(self.app.network, f)
        f.close()
    
    # Checks for the escape button to go back to start mode
    def keyPressed(self, event):
        if event.key == "Escape":
            self.app.setActiveMode(self.app.startMode)
    
    # Called when the mouse is pressed, checks for button presses
    def mousePressed(self, event):
        point = (event.x, event.y)
        for panel in self.panels:
            bounds = panel.getBounds()
            if pointInBounds(point, bounds):
                panel.mousePressed(point)
                return
    
    # Generates a 2d list of the confusion maatrix with rows representing
    # predicted class and columns representing the actual (true) class.
    def generateConfusionMatrix(self):
        net = self.app.network
        results = [(net.forwardPropagation(x), y) for (x, y) in self.testData]
        matrix = make2dList(self.numLabels, self.numLabels)
        for predicted, actual in results:
            winningLabelIndex = None
            highestPercentage = -1
            for i in range(len(predicted)):
                if predicted[i][0] > highestPercentage:
                    highestPercentage = predicted[i][0]
                    winningLabelIndex = i
            # Test against true label
            actualLabelIndex = np.argmax(actual)
            matrix[winningLabelIndex, actualLabelIndex] += 1
            cellVal = matrix[winningLabelIndex][actualLabelIndex]  
            if cellVal > self.maxMarginalCount:
                self.maxMarginalCount = cellVal
        self.confusionMatrix = matrix
    
    # Constructs and draws the confusion matrix onto the canvas with a
    # gradated color legend
    def drawConfusionMatrix(self, canvas, x, y, width, height):
        dRow = height/self.numLabels        # Change in y value between rows
        dCol = width/self.numLabels         # Change in x value betwen cols
        # Titles
        canvas.create_text(x, y + height/2, text = "Predicted Class\n\n",
                           angle = 90, anchor = "s", font = "Arial 9 bold")
        canvas.create_text(x + width/2, y, text = "Actual Class\n\n",
                           anchor = "s", font = "Arial 9 bold")
        canvas.create_text(x + width/2, y - 20, text = "Confusion Matrix\n\n",
                           font = "Arial 11 bold", anchor = "s")
        maxCount = self.maxMarginalCount

        # Create and fill squares with text and shading
        for row in range(self.numLabels):
            rowY = dRow*(row) + y
            nextRowY = dRow*(row + 1) + y
            midY = (rowY + nextRowY) / 2
            canvas.create_line(x, rowY, x + width, rowY)
            rowLabel = self.app.data.labels[row]
            canvas.create_text(x, midY, text = rowLabel, angle = 90,
                               anchor = "s")
            for col in range(self.numLabels):
                colX = dCol*(col) + x
                nextColX = dCol*(col + 1) + x
                midX = (colX + nextColX) / 2
                canvas.create_line(colX, y, colX, y + height)
                # Draws column label
                colLabel = self.app.data.labels[col]
                canvas.create_text(midX, y, text = colLabel, anchor = "s")
                # Shades the cell according to its relative magnitude, with
                # respect to the highest magnitude cell
                count = self.confusionMatrix[row][col]
                fill = mapPercentToLegendColor(count / (maxCount + 1),
                                               self.RGB1, self.RGB2)
                canvas.create_rectangle(colX, rowY, nextColX, nextRowY,
                                        fill = fill)
                # The percentage of the whole dataset that this cell represents
                marginalPercentString = "%0.2f" % (count / self.numExamples)
                cellText = f'{count} ({marginalPercentString})'
                canvas.create_text(midX, midY, text = cellText)
        
        # Color legend
        legendXOffset = self.app.width / 100
        legendWidth = self.app.width / 40
        drawColorGradientVertical(canvas, x + width + legendXOffset, y,
                                  legendWidth, height, self.RGB2, self.RGB1)
        # Tick marks
        numTicks = 6
        dY = height / numTicks
        dCount = maxCount / numTicks
        tickX = width + legendWidth + legendXOffset + x
        for i in range(numTicks):
            tickY = y + height - dY*i
            canvas.create_line(tickX, tickY, tickX + 5, tickY)
            canvas.create_text(tickX + 5, tickY, text = int(dCount*i),
                               anchor = "w")

    # Draws the performance measures onto the canvas
    def drawPerformanceMeasures(self, canvas, x, y):
        canvas.create_text(x, y, text = "Performance Measures",
                           font = "Arial 12 bold", anchor = "nw")
        
        precisionString = f'Precision: {self.precisionFormatted}\n'
        recallString = f'Recall: {self.recallFormatted}\n'
        f1ScoreString = f'F1: {self.f1ScoreFormatted}\n'
        performanceSummary = precisionString + recallString + f1ScoreString
        canvas.create_text(x, y + 30, text = performanceSummary, anchor = "nw",
                           font = "Arial 10")
    
    # Based on formulas in the ICML 2004 Notes on classification performance
    # metrics: http://people.cs.bris.ac.uk/~flach/ICML04tutorial/ 
    # Returns the precision as calculated by TP_avg / (TP_avg + FP_avg)
    def calculatePrecision(self):
        # Calculate sum of true positives for all classes
        # Calculate sum of false positives for all classes
        sumOfPrecisions = 0
        for classIndex in range(self.numLabels):
            sumOfPrecisions += self.calculatePrecisionForClass(classIndex)
        avgPrecision = sumOfPrecisions / self.numLabels
        return avgPrecision
    
    # Calculates the precision for the specified class index in the confusion
    # matrix
    def calculatePrecisionForClass(self, classIndex):
        truePositives = self.confusionMatrix[classIndex][classIndex]
        falsePositives = 0
        for col in range(len(self.confusionMatrix)):
            if col == classIndex: continue
            falsePositives += self.confusionMatrix[classIndex][col]
        if truePositives + falsePositives == 0:
            return 0
        precision = truePositives / (truePositives + falsePositives)
        return precision

    # Based on formulas in the ICML 2004 Notes on classification performance
    # metrics: http://people.cs.bris.ac.uk/~flach/ICML04tutorial/
    # Calculates recall as calculated by TP_avg / (FN_avg + TP_avg)
    def calculateRecall(self):
        sumOfRecalls = 0
        for classIndex in range(self.numLabels):
            sumOfRecalls += self.calculateRecallForClass(classIndex)
        avgRecall = sumOfRecalls / self.numLabels
        return avgRecall

    # Calculates the recall for the specified class index in the confusion
    # matrix
    def calculateRecallForClass(self, classIndex):
        truePositives = self.confusionMatrix[classIndex][classIndex]
        falseNegatives = 0
        for row in range(len(self.confusionMatrix)):
            if row == classIndex: continue
            falseNegatives += self.confusionMatrix[row][classIndex]
        if truePositives + falseNegatives == 0:
            return 0
        recall = truePositives / (truePositives + falseNegatives)
        return recall

    # Based on formulas in the ICML 2004 Notes on classification performance
    # metrics: http://people.cs.bris.ac.uk/~flach/ICML04tutorial/ 
    # Calculates F1 score based on 2*PRECISION * RECALL / (PRECISION + RECALL)
    def calculateF1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)
        
    def redrawAll(self, canvas):
        matrixHeight = matrixWidth = self.app.height / 2
        topX, topY = self.app.width / 7, self.app.height / 5
        self.drawConfusionMatrix(canvas, topX, topY,
                                 matrixWidth, matrixHeight)
        topX, topY = self.app.width / 2, self.app.height / 3
        self.drawPerformanceMeasures(canvas, topX, topY)
        self.app.drawPanels(canvas, self.panels)
        self.mainPanel.drawPanelVertical(canvas)

class NeuralNetworkApp(ModalApp):
    ACTIVATION_FUNCTION_NAMES = ["Logistic", "TanH"]
    ACTIVATION_FUNCTIONS = {"Logistic" : logistic, "TanH" : tanH}
    NODE_RADIUS_RATIO = 1/40
    # The ceiling above which parameter values will not be further distinguished
    # by color 
    COLORIZATION_BOUND = 3

    # Starts the Neural Network App
    def appStarted(self):
        self.oldWidth = self.width
        self.oldHeight = self.height
        self.debug = False
        self.margin = 50
        self.datasetIndex = 0
        self.activationFunctionIndex = 0
        self.allPanels = []
        self.datasets = self.findAllDatasetsInDirectory()
        self.data = self.datasets[0]
        self.alpha = 1
        self.defaultParameters = {'dims' : [4, 5, 5, 2],
                                  'activation' : self.activationFunctionIndex,
                                  'dataset' : self.datasetIndex}
        self.initNetwork()
        self.startMode = StartMode()
        self.configMode = ConfigMode()
        self.trainMode = TrainMode()
        self.testMode = TestMode()
        self.setActiveMode(self.startMode)
    
    # Initializes a background display-network (only a prop, cannot be trained)
    def initDisplayPieceNetwork(self):
        dims = [4, 7, 6, 1, 1, 6, 7, 4]
        self.network = NeuralNetwork(dims, None)
        self.updateNetworkViewModel()

    # Initializes a network for training using the default network parameters
    def initNetwork(self):
        dims = copy.copy(self.defaultParameters['dims'])
        funcName = self.ACTIVATION_FUNCTION_NAMES[self.defaultParameters['activation']]
        func = self.ACTIVATION_FUNCTIONS[funcName]
        self.network = NeuralNetwork(dims, func)
        self.updateNetworkViewModel()
    
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
        self.updateNetworkViewModel()

    # Finds all CSV files in the working directory and returns a list of them
    def findAllDatasetsInDirectory(self):
        filepaths = listFiles('datasets', suffix = '.csv')
        datasetList = []
        for path in filepaths:
            newDataset = Dataset(path)
            datasetList.append(newDataset)
        return datasetList

    # Normalizes the weight between 0 and 1 and returns
    # mapPercentToLegendColor(%, rgb1, rgb2) i.e., converts parameter to 
    # a color and returns RGB value
    def weightToColor(self, weight, rgb1, rgb2, biasTerm = False):
        maxWeight = self.COLORIZATION_BOUND
        minWeight = -maxWeight
        weightRange = 2*maxWeight
        normalizedWeight = (weight + maxWeight) / weightRange
        percent = getInBounds(normalizedWeight, 0, 1)
        return mapPercentToLegendColor(percent, rgb1, rgb2)

    # Draws current dataset and activation function name
    def drawConfigInfo(self, canvas):
        s = f'Dataset: {self.data}\n'
        # Special attribute __name__ used as described in Python 3 docs:
        # https://docs.python.org/3/library/stdtypes.html#special-attributes
        funcName = self.network.activation.__name__
        s += f'Activation Function: {funcName}'
        canvas.create_text(self.width - self.margin, self.height - self.margin,
                            anchor = "se",
                            text = s)

    # Refresh network view model on window resize with updateNetworkViewModel()
    def sizeChanged(self):
        self.updateAllPanelViewModels()
        self.updateNetworkViewModel()
        # FIXME: For some reason, resizing the window weirdly affects the hover
        #        mode in TrainMode, but this if statement undoes that change.
        #        Need to find the cause of this bug though.
        if self._activeMode == self.trainMode:
            self.trainMode.toggleHoveringMode()
            self.trainMode.toggleHoveringMode()

        self.oldWidth = self.width
        self.oldHeight = self.height
    
    # Updates the view models for panels in all modes
    def updateAllPanelViewModels(self):
        for panelsInMode in self.allPanels:
            for panel in panelsInMode:
                panel.sizeChanged(self.width, self.height,
                                   self.oldWidth, self.oldHeight)
        
    # Recalculate node radius and network view coordinates
    def updateNetworkViewModel(self):
        self.r = min(self.width*self.NODE_RADIUS_RATIO,
                     self.height*self.NODE_RADIUS_RATIO)
        self.regenerateNodeCoordinates()
        self.regenerateNetworkViewBounds()
        self.nodeCoordinatesSet = set(flatten2dList(self.nodeCoordinates))

    # Draws the bias for a specified layer and node
    def drawBias(self, canvas, l, n, coords, r, visualizeParams, visualizeMe,
                 rgb1, rgb2):
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
            bColor = self.weightToColor(bi, rgb1, rgb2, biasTerm = True)
        cx, cy = coords[l][n]
        canvas.create_oval(cx-r, cy-r, cx+r, cy+r, width = biMagnitude,
                           fill = bColor)
        
    # Draws the weights for a specified layer and node
    # visualizeMe takes precedence over visualizeParams
    def drawWeights(self, canvas, l, n, coords, r, visualizeParams, visualizeMe,
                   doStipple, rgb1, rgb2):
        cx, cy = coords[l][n]
        if l == len(self.network.dims) - 1:
            return
        for n2 in range(len(coords[l+1])):
            if visualizeMe:
                wij = self.network.w[l][n2][n]
                wColor = self.weightToColor(wij, rgb1, rgb2)
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
    def drawNetwork(self, canvas, visualizeParams = True, doStipple = True,
                    rgb1 = "255255255", rgb2 = "255255255"):
        visualizeMe = True # default
        r = self.r  # Node radius
        coords = self.nodeCoordinates
        for l in range(len(coords)):
            for n in range(len(coords[l])):
                if visualizeParams and self._activeMode != self.startMode:
                    xy = coords[l][n]
                    visualizeMe = xy in self.trainMode.selectedNodeCoords
                elif self._activeMode != self.startMode:
                    visualizeMe = False
                self.drawBias(canvas, l, n, coords, r,
                                visualizeParams, visualizeMe, rgb1, rgb2)

                self.drawWeights(canvas, l, n, coords, r, visualizeParams,
                                 visualizeMe, doStipple, rgb1, rgb2)

    # Call this to update the view-model representation of the network's
    # individual nodes. This method populates a jagged 2d list L where L[i][j]
    # contains a tuple (x, y) of the coordinates of the jth node in the ith
    # layer of the network.
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
    
    # Call this to update the view-model representation of the network's
    # outermost bounds B on the canvas, where B is a 4-tuple containing
    # the points on the corner of the rectangle circumscribing the network
    def regenerateNetworkViewBounds(self):
        coords = self.nodeCoordinates
        x = [point[0] for layer in coords for point in layer]
        y = [point[1] for layer in coords for point in layer]
        ax1, ax2 = min(x) - self.r, max(x) + self.r
        ay1, ay2 = min(y) - self.r, max(y) + self.r
        self.networkViewBounds = (ax1, ay1, ax2, ay2)
    
    # Given a list of panels and a canvas, draws all the panels onto that canvas
    def drawPanels(self, canvas, panels):
        for panel in panels:
            panel.drawPanelVertical(canvas)
    
    # Given a button title and a mode, configures a panel for that mode with a 
    # backButton button for forward navigation. Requires the target mode to have
    # a goNext function specifying the target mode and exit behavior.
    def configureNextPanel(self, name, mode):
        width = self.width / 10
        height = self.height / 30
        mode.nextPanel = Panel(self.width - self.margin, self.margin,
                          width, height, anchor = "ne")
        mode.nextButton = Button(mode.goNext, name)
        mode.nextPanel.addButton(mode.nextButton)
        mode.panels.append(mode.nextPanel)
    
    # Given a button title and a mode, configures a panel for that mode with
    # a backButton button for backwards navigation. Requires the target mode
    # to have a goBack function specifying the target mode and exit behavior.
    def configureBackPanel(self, name, mode):
        width = self.width / 10
        height = self.height / 30
        mode.backPanel = Panel(self.margin, self.margin, width, height,
                          anchor = "nw")
        mode.backButton = Button(mode.goBack, name)
        mode.backPanel.addButton(mode.backButton)
        mode.panels.append(mode.backPanel)
    
    # Given a mouse click point and a list of panels, checks if the user clicked
    # on any buttons within any of the panels. If so, activate that button and
    # return True, otherwise return False.
    def doPanelPress(self, point, panels):
        for panel in panels:
            bounds = panel.getBounds()
            if pointInBounds(point, bounds):
                panel.mousePressed(point)
                return True
        return False

if __name__ == "__main__":
    import cProfile, pstats
    cProfile.run("NeuralNetworkApp(width = 1450, height = 800)", f'{__file__}.profile')
    s = pstats.Stats(f'{__file__}.profile')
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)
    #NeuralNetworkApp(width = 1700, height = 900)