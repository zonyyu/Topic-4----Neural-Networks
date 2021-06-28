# Topic 4 -- Neural Networks

## Overview of this Repository
This repository contains all the teaching material related to **Neural Networks**. The master `branch` contains the sample code for the instructor to **reference**, and the `workshop` branch contains the **empty notebooks** for the instructor and students to program in.

## For the Instructors:

This section details instructions to guide the instructor in delivering the course. The instructor should fill out the blank notebooks in the `workshop` branch according to the reference in the `master` branch (if it is possible, use two monitors so you can have the reference code opened side by side.) **There will be function calls already written in the blank notebook, please run those calls without modifications.**

Below is the curriculum of this repository, as well as the order of content to be delivered. **Be sure to familiarize yourself with the code before teaching! Feel free to explore the notebooks for course material as well as the programming exercise `lock.py`.**

This document will detail the content of material to be delivered **in order**.

### Getting set up (Jupyter Notebok)
1. Clone this repo into a working directory
2. Switch to the `workshop` branch:
   ```bash
    $ git checkout workshop
   ```
3. Create and activate a virtual environment with the command below:
    ```bash
    # MacOS/Linux
    $ python3 -m venv env 
    $ source env/bin/activate
    ```
    ```bat
    :: Windows
    \> python -m venv env 
    \> .\env\Scripts\activate
    ```
4. If you are in the virtual environment, you should see the `(env)` marker. Now, install all the dependencies:
    ```bash
    # MacOS/Linux
    $ pip install -r requirements.txt
    ```
    ```bat
    :: Windows
    \> python -m pip install -r requirements.txt
    ```
5. You are ready to go!

### Getting Started (Google Colab)
1. Clone this repo into your working directory
2. Switch to the `workshop` branch
    ```bash
    $ git checkout workshop
    ```
3. Upload whichever notebook you need to work on into Colab.
4. Drag the `colab.zip` file into Colab.
5. Unzip the file and install the dependencies using `pip` within Colab.
6. You're ready to go!


## Topic 4 -- Neural Networks

### Installing Dependencies
- Talk a bit about the dependencies you're using

### Logistic Regression vs Neural Networks

- Paraphrase the text for the general idea
- Recap the students with now Logistic Regression works.
  - Linear sum + sigmoid activation
- Neuron Representation of Logistic Regression
  - Explain the different parts of the diagram
  - Follow the text to get an idea of what to talk about.
  - Maybe explain the weights $w$ as the strength of connection between neurons.
  - Multi-class classification using a neuron representation
- Stacking Logistic Regression Units
  - Output of one Logistic Regression layer used as the input of another
  - It can learn much more complicated hypothesis functions


### Neural Networks
- Layers
  - Talk about how to count layers
  - Input layer
  - Output layer
    - Decides on the problem type
  - Hidden layers
    - Dramatically increases the complexity of tasks the NN can learn.

- Activation Functions
  - Talk about how using sigmoid for the hidden layers results in slow training
  - Tanh activation
  - ReLU activation
  - Remind students that sigmoid is still used on the output layer if your NN is set up for classification.

- Why we need activation functions
  - The problem of having linear activations in the hidden layers is that your NN will behave as if it has a single layer.
  - Skip the math part

### How Neural Networks Learn

- Reiterate the similarities with how Linear/Logistic regression models learn.
- When describing, refer to the images.
- Feel free to refer to the text

### Neural Network Applications

- Mention that Regression and Binary Classification is very similar to what we learned before, thus we are not going to talk about it
- Multi-class Classification
  - for each example, NN predicts on 1 class.
  - Recall Multi-class Classification that was covered before.
  - Talk a bit about the example layed out in the text
  - Softmax Activation
    - Be sure to mention about the output probabilities summing to 1
    - Don't worry about the math

  - Categorical Cross Entropy
    - Loss function that is used with softmax activation

- Multi-task Classification
  - Be sure to give the self driving car example in the text.
  - Explain the difference between multi-class and multi-task learning


### Neural Networks in Action

- Paraphrase the text when teaching
- Creating the Ultimate Candy
  - A fun scenario created to entertain the students

- Visualizing the Dataset
  - Make sure students understand what each line of code does
  - The goal is to predict whether or not the candy will have a certain feature (chocolate, fruity, caramel, ...) given the inputs of sugar percent and win percent
  - Ask students whether this is Regression, Binary classification, Multi-class classification, or Multi-task classification

- Training the Neural Network
  - Paraphrase the text when teaching
  - Commentate while coding. The comments in the code may give you a clue for what to talk about
  - Question for students:
    - Should we use linear activation on the hidden layers?

  - **You should obtain a fairly low accuracy. if your accuracy is below 0.1818, maybe rerun the cells to create the training and testing sets and train again.**
    - This is because the dataset is very small, and sometimes `train_test_split` can shuffle the data in a way that hinders performance.

- How Accuracy is Calculated in SKLearn
  - paraphrase from the text
  - Be sure that students understand how accuracy should be calculated differently for Multi-task and multi-class


- Other Metrics
  - Paraphrase from the text
  - Make sure students understand what is going on in the code
  - Notice that many of the classes will have an F1 Score of 0.
    - This motivates us to make a confusion matrix for each class

  - When making the confusion matrix, make sure students understand what is going on
  - Analyze the confusion matrix. the classes with an F1 score of 0 are the ones without True Positives
  - Discuss with students where your network does well, and where it doesn't do too well 


- Conclusion
  - Conclude that your network is accurate for some classes, and less accurate for other classes
  - paraphrase the text
  - Try with relative cost of 0.1 and popularity of 90
  - Ask the students
    - What might cause the poor performance in this Neural Network.

## About the Programming Exercise
- There are three files:
  - `nn.py`, `lock.py`, and `mnist.ipynb`

- `lock.py` runs a GUI application that allows the user to draw in the boxes
- `nn.py` will require you to copy over your `NN()` class and `CCE()` function
  - `nn.py` already contains the skeleton for the `predict_num()` function, which has to be filled out.
- `mnist.ipynb` is used to train the neural network.
  - fill this notebook out, commentate while coding 
  - use the comments in the code to guide you in your commentary.


### Procedures
- first complete `mnist.ipynb`
  - fill this notebook out, commentate while coding 
  - use the comments in the code to guide you in your commentary.
  - If you feel the need to help students visualize by drawing in OneNote, do it. 
  - at the end, save the model as `mnist_predictor.pt`
  - if trained in google colab, be sure to download the model
  
- open `nn.py`
  - Copy your code for the `NN()` class as well as the `CCE()` function into `nn.py`
  - write code to load the model in
  - fill out `predict_nums()` to process the input images and output prediction

- run `lock.py` to see your neural network in action





