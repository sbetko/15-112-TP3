import os
from mymathlib import *

# Object representing a CSV formatted dataset. Expected delimiter is a comma
# unless optionally specified and headers are not supported. The final column is 
# assumed to be the output variable.

# The dataset class is only sufficient for prediction problems
# involving quantitative features and categorical labels. I.e.,
# single- or multi-class classification with numerical inputs.
# This is because the implementation of this class ONLY supports
# quantitative feature values and does not provide support for
# converting quantitative inputs to one-hot encodings. 

class Dataset(object):
    TRAIN_SPLIT = 0.6
    TEST_SPLIT = 0.2
    VALIDATION_SPLIT = 0.2

    # Initializes the dataset object
    def __init__(self, path, delim = ','):
        self.path = path
        print(path)
        self.train = []
        self.validation = []
        self.test = []
        self.labels = []
        self.features = []
        self.name = self.getNameFromFilepath()
        self.delim = delim
        self.initializeData()

    # Returns the filename without extension from the filepath
    def getNameFromFilepath(self):
        myFilename = os.path.split(self.path)[-1]
        myDatasetName = myFilename.split(".")[0]
        return myDatasetName
    
    # Returns a string representation of the dataset
    def __repr__(self):
        return f'{self.name} {self.numFeatures}x{self.numLabels}'

    # Reads dataset from CSV file and builds dataset with one-hot index
    def buildDataset(self):
        result = []
        with open(self.path) as csv:
            for line in csv:
                # Skip empty lines
                if line.strip() == "":
                    continue
                row = line.split(self.delim)
                x = []
                # Do not include last column (label) in feature set
                for feature in row[:-1]:
                    x.append([float(feature)])
                
                # Last column in CSV is the output label
                y = self.convertLabelToOneHot(row[-1].strip())
                xyTuple = (x, y)
                result.append(xyTuple)
        return result

    # Initializes the instances data attributes
    def initializeData(self):
        self.allData = []
        # Must figure how many labels there are before converting them to
        # one-hot later in a second pass.
        labelSet = set()
        with open(self.path) as csv:
            for row in csv:
                cleanRow = row.strip()
                if cleanRow == "":
                    continue
                label = cleanRow.split(self.delim)[-1]
                labelSet.add(label)
                
        # Find labels and build a dictionary to map labels to one-hot indices
        self.labels = sorted(list(labelSet))
        self.numLabels = len(self.labels)
        self.oneHotIndexDictionary = dict()
        for labelIndex in range(self.numLabels):
            self.oneHotIndexDictionary[self.labels[labelIndex]] = labelIndex
        
        # Perform second pass to build dataset now that we have one-hot labels
        self.allData = self.buildDataset()
        
        self.numFeatures = len(self.allData[0][0])
        random.shuffle(self.allData)
        splitTrainAndVal = int(len(self.allData)*self.TRAIN_SPLIT)
        splitValAndTest = splitTrainAndVal + int(len(self.allData)*self.VALIDATION_SPLIT)
        self.train = self.allData[:splitTrainAndVal]
        self.validation = self.allData[splitTrainAndVal:splitValAndTest]
        self.test = self.allData[splitValAndTest:]

    # Builds one-hot vector for output label given a one-hot dictionary
    # One-hot is a vector encoding of fixed length with only one component
    # set to 1 (True) and the rest set to 0 (False).
    def convertLabelToOneHot(self, label):
        y = make2dList(len(self.labels), 1)
        myOneHotIndex = self.oneHotIndexDictionary[label]
        y[myOneHotIndex] = [1]
        return y