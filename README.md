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
Is the code functional? | Yes, the code is functional.
The model provided can be used to successfully operate the simulation. | Yes, the model and the drive.py file successfully operate the simulation.
Is the code usable and readable? | Yes.

The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.
My last model code is in Train-the-model.ipynb. Left only the functional code for the final training.

## Model Architecture and Training Strategy

Question | Answer
------ | -------
Has an appropriate model architecture been employed for the task? | Yes. - 2 convolutional layers and 3 dense layers.
The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model. | Yes. Data is normalized by `x / 255 - 0.5`.
Has an attempt been made to reduce overfitting of the model? | Yes. Overfiting was achieved using CV set and dropouts. After appropriate model's architecture found and paraters optimized - removed the CV to use all data for training.
Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.  | Yes.
Have the model parameters been tuned appropriately?  | Yes - manualy after many hours of experimentation. 
Learning rate parameters are chosen with explanation, or an Adam optimizer is used.  | Adam
Is the training data chosen appropriately?  | Made many training sets iterations, including recovery data.
Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).  | Yes

## Architecture and Training Documentation

Question | Answer
------ | -------
Is the solution design documented? | Yes, at bottom.
The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem. | Yes, at bottom.
Is the model architecture documented? | Yes.
The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged. | Yes.
Is the creation of the training dataset and training process documented? | Yes.
The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included. | Yes.

## Simulation

Question | Answer
------ | -------
Is the car able to navigate correctly on test data? | Yes.
No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). | Yes.



