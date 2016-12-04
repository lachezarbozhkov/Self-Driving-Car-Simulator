# Self-Driving-Car-Simulator
Deep Learning Behaviour Cloning for Self Driving Car Simulator

## Use Deep Learning to Clone Driving Behavior

Question | Answer
------ | -------
Is the code functional? | 
The model provided can be used to successfully operate the simulation. | 
Is the code usable and readable? |

The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

## Model Architecture and Training Strategy

Question | Answer
------ | -------
Has an appropriate model architecture been employed for the task? | Not yet
The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model. | Not yet
Has an attempt been made to reduce overfitting of the model? | Not yet
Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.  | Not yet
Have the model parameters been tuned appropriately?  | Not yet
Learning rate parameters are chosen with explanation, or an Adam optimizer is used.  | Not yet
Is the training data chosen appropriately?  | Not yet
Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).  | Not yet

## Architecture and Training Documentation

Question | Answer
------ | -------
Is the solution design documented? | Not yet
The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem. | Not yet
Is the model architecture documented? | Not yet
The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged. | Not yet
Is the creation of the training dataset and training process documented? | Not yet
The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included. | Not yet

## Simulation

Question | Answer
------ | -------
Is the car able to navigate correctly on test data? | Not yet
No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). | Not yet
