### Project Title
Shallow Neural Network for Mushroom Classification
Orobosa Igbinovia

### Programming Language Requirement

This project was implemented and tested using Python 3.
All scripts, syntax, and runtime behavior assume Python 3.x — running this project with Python 2 will not work.

### Project Description

This project implements a shallow feed-forward neural network trained via backpropagation to classify mushrooms in the Agaricus–Lepiota dataset as edible or poisonous. The classifier is entirely implemented from scratch in Python 3, without external machine learning libraries.
The network consists of:
Input layer (one-hot encoded attributes)
Hidden layer with 6 neurons
Output layer with 2 neurons representing classes
[1, 0] = edible, [0, 1] = poisonous
The network uses:
Sigmoid activation
Squared error loss
Backward gradient propagation using the chain rule

### Files Included

Fil----------------------Purpose

formulas.py--------------Sigmoid functions, squared error computation, derivatives
models.py----------------Layer class, forward propagation, backpropagation logic
proj_test.py-------------End-to-end driver script for training / validation / testing
Mushroom_converter.py----Converts raw dataset into one-hot encoded vectors
training.txt, val.txt, testing.txt	Final dataset splits

### *_err.txt	Error logs generated during execution

Data Processing
The raw dataset is converted from categorical entries into binary vectors:
First 2 entries = class vector
'e' → [1, 0]   (edible)
'p' → [0, 1]   (poisonous)
Each remaining feature is one-hot encoded based on attribute values defined in agaricus-lepiota.names
The missing value '?' under stalk-root is treated as a valid category
Dataset split:
Set	Percentage	Range	Count
Training	70%	0–5685	5686
Validation	15%	5686–7107	1422
Testing	15%	7108–8123	1016
Neural Network Architecture
Input → Hidden Layer (6 neurons) → Output Layer (2 neurons)
Activation: Sigmoid
Loss Function: 0.5 * Σ (t − y)²
Learning Rate: 1.0

### Major Issues Encountered and Fixes

1. Incorrect Output Gradient Sign

Problem:
Used (output - target) instead of (target - output).
Effect:
Network got trapped at 46.9% accuracy due to gradient ascent rather than descent.
Fix:
Corrected sign inside output-layer backprop.
Result:
Training error decreased properly and accuracy exceeded 99%.

2. Error Computed Using Only First Output Neuron

Fix:
Replaced partial error calculation with:
err(y.layer_out, target[curr_point])
Result:
Accurate loss and classification reporting.

3. Incorrect One-Hot Encoding Order

Attributes were encoded in dictionary order instead of dataset column order.
Fix:
Forced strict dataset-based ordering.

4. Missing Value Handling

Added '?' as valid category so vector size remains consistent.

5. Hidden Layer Backprop Using Stale Activations

Fix:
Ensured layers update input vectors before evaluation.

6. Bimodal Accuracy Behavior

Faulty gradient → 46.9% plateau
Correct gradient → ~99% convergence
Once fixed, the network achieved stable high accuracy.

Final Results
Dataset	Error	Accuracy
Training	Very low	≈ 99%
Validation	Very low	≈ 98–99%
Testing	Very low	≈ 99%

### How to Run

Ensure Python 3.x is installed, then run:
py proj_test.py or python proj_test.py

This will:
Train the network
Validate the model
Test on unseen data
Generate:
training_err.txt
val_err.txt
testing_err.txt

### Partial Result
-----------------------Training output---------------------------

Data has converged at iteration 30400
Neural network is done training!
Error percentage on training set: 0.00852

-----------------------Validation output---------------------------

Validation complete.
Error percentage on validation set: 6e-05
Model Accuracy: 0.99994

-----------------------Testing output---------------------------

Testing done! Check generated output files.
Error percentage on testing set: 0.003937007874015748
Model Accuracy: 0.9960629921259843

### Acknowledgment of Discussions
This is a single-person project. I did not discuss the implementation with any other students.

### Project Status
All components of this project work as intended. The network trains, validates, and tests correctly, achieving ~99% accuracy.
