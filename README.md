# 15-112-TP3
Interactive Visualization of the Fully Connected Multilayer Perceptron Model.

This project provides a visual demonstration of how one type of neural network—the multilayer perceptron (MLP)—learns to approximate functions using the gradient descent and backpropagation algorithms. It also provides tools for designing your own neural network structure, functionality for importing and exporting trained models, performance evaluation, and the use of custom datasets.

---------------------------------------------------------------------------

Please run the file "neuralnetworkapp.py" in your editor to start the animation. You may include any .CSV file in the \datasets folder in the project directory to be read and imported as a dataset. The program's CSV reader expects comma delimiters and requires there be no header. Additionally, all columns must contain quantitative values except for the final column, which is qualitative. This is a restriction of the Dataset class used which currently only supports datasets with quantitative explanatory variables and categorical response variables.

---------------------------------------------------------------------------

The animations in this application are built using the CMU 15-112 Animation Framework, version 0.8.5. A copy of the animation framework has been included with the source files, but you can download a copy of the framework here: https://www.cs.cmu.edu/~112/notes/cmu_112_graphics.py

This project also requires the pickle module, a standard python module for object serialization. It should usually be included in your python distribution, but a copy has been included with the source files. You can acquire this library here: https://docs.python.org/3/library/pickle.html

---------------------------------------------------------------------------

Shortcuts:
In Train Mode

                Press space to start or pause training.
		
                Press the right arrow key to skip forward 1 iteration
		
                Press r to reset weights and biases.
		
                Press t to change visualization mode.
		
                Press up or down to increase or decrease the learning rate.
		
                Press enter to test the model.
		
                Press escape to go back to configuration mode.

In Configuration Mode


		Press right and left arrow keys to add and remove layers.
		
             	Press up and down arrow keys to add and remove neurons.
		
             	Press a to change activation function.
		
             	Press tab to change datasets.
		
             	Press r for default settings.
		
             	Press space to begin training.
		
