# Self-Driving-Car-Simulator
Deep Learning Behaviour Cloning for Self Driving Car Simulator
[Udacity Self Driving Car Engineer Nanodegree](https://www.udacity.com/drive)


This README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and video of the images from the train dataset and the simulations are included.	

## Creation of the training dataset and training process documented
The udacity self-driving-car simulator was used to generate train and CV image sets including the steering angle. <br/>
For generating the dataset I played the game on the first track only. The 640x480 resolution of the game was used with lowest quality details. When in training mode the game produces images from _center_, _left_, and _right_ angle cameras and the  angle postion of the steering wheel. Writing the image paths and steering angle to a csv file.
For the train set I used the my own recorded data from track one, including _recovery data_. Also used the training data provided from _udacity_. Used the _left_ and _right_ images with manually augmented steering angle from the images, except recovery data for aditional train data - the model learned to aim for centering between the lanes. <br/>
The dataset preparaion can be found in [Prepare Train Data.ipynb](Prepare%20Train%20Data.ipynb)

Total number of generated images for training is around 55k. <br/>
You can see a video of the full training set, with added steering angle in the following video: <br/>
[https://www.youtube.com/watch?v=3qQvYdbr8PM](https://www.youtube.com/watch?v=3qQvYdbr8PM)

[![ScreenShot](http://img.youtube.com/vi/3qQvYdbr8PM/0.jpg)](https://www.youtube.com/watch?v=3qQvYdbr8PM)





## Solution design


## Approach taken


## Model architecture






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

The code in Train-the-model.ipynb uses a Python generator, to generate data for training rather than storing the training data in memory. The Train-the-model.ipynb code is clearly organized and comments are included where needed.
My last model code is in Train-the-model.ipynb. Left only the functional code for the final training.

## Model Architecture and Training Strategy

Question | Answer
------ | -------
Has an appropriate model architecture been employed for the task? | Yes.
The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model. | Yes. Data is normalized by `x / 255 - 0.5`.
Has an attempt been made to reduce overfitting of the model? | Yes. Overfiting was achieved using CV set and dropouts. 
Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting.  | Yes.
Have the model parameters been tuned appropriately?  | Yes - manualy after many hours of experimentation. No automatic parameter serach performed.
Learning rate parameters are chosen with explanation, or an Adam optimizer is used.  | Adam
Is the training data chosen appropriately?  | Made many training sets iterations, including recovery data.
Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).  | Yes

## Architecture and Training Documentation

Question | Answer
------ | -------
Is the solution design documented? | Yes, at top.
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

## Install the env:

* Install anaconda python 3.5
* Install packeges: 
- tensorflow
- numpy
- flask-socketio
- eventlet
- pillow
- keras
- h5py
* Install simulator from links above.
* Play and generate training data.
* conda install -c menpo opencv3
* pip install moviepy

