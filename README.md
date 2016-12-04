# Self-Driving-Car-Simulator
Deep Learning Behaviour Cloning for Self Driving Car Simulator

## Download game simulator:
<p>We’ve created a simulator for you based on the Unity engine that uses real game physics to create a close approximation to real driving.</p>
<p>Download it here:</p>
<ul>
<li><a href="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip" target="_blank">Linux</a></li>
<li><a href="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip" target="_blank">macOS</a></li>
<li><a href="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip" target="_blank">Windows 32-bit</a></li>
<li><a href="https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip" target="_blank">Windows 64-bit</a></li>
</ul>
<h4 id="running-the-simulator">Running the Simulator</h4>
<p>Once you’ve downloaded it, extract it and run it.</p>
<p>When you first run the simulator, you’ll see a configuration screen asking what size and graphical quality you would like. We suggest running at the smallest size and the fastest graphical quality. 
We also suggest closing most other applications (especially graphically intensive applications) on your computer, so that your machine can devote its resource to running the simulator.</p>


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
