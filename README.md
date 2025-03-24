# HeartDiseasePredictionModel

Heart Disease Prediction

A simple neural network using PyTorch to predict heart disease based on patient data. The model is trained on the UCI Heart Disease Dataset and achieves 85.25% accuracy.

## Features

	-	Loads and processes heart disease data
	-	Selects the top 8 features for better accuracy
	-	Uses a 4-layer neural network with batch normalization and dropout
	-	Optimized training with SGD, learning rate scheduling, and early stopping

Installation
	1.	Clone the repository:

    git clone https://github.com/your-username/heart-disease-prediction.git
    cd heart-disease-prediction


2.	Install dependencies:

        pip install numpy torch matplotlib scikit-learn


3.	Run the script:

         python heart_disease_nn.py



Usage
	- The script downloads the dataset automatically.
	- The model trains on 80% of the data and tests on 20%.
	-	Training progress and loss values are printed.
	-	At the end, test accuracy (85.25%) is displayed.
	-	A loss graph is plotted.

Model Overview
	-	Input: 8 selected features
	-	Hidden layers:
	-	256 neurons → ReLU → Dropout
	-	128 neurons → ReLU → Dropout
	-	64 neurons → ReLU → Dropout
	-	Output: 1 neuron (Sigmoid activation)

