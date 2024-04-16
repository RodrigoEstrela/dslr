import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler


def sigmoid(z):
    """Sigmoid function."""
    return expit(z)


def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """Gradient Descent to optimize theta."""
    m = len(y)
    for _ in range(num_iterations):
        h = sigmoid(theta @ X.T)
        gradient = X.T @ (h - y) / m
        theta -= learning_rate * gradient
    return theta

def train_logistic_regression(X, y, num_classes, learning_rate, num_iterations):
    """Train logistic regression models for each class using one-vs-all strategy."""
    theta_models = {}

    for i in range(num_classes):
        # Convert y to binary for the current class
        y_binary = (y == i).astype(int)
        # Initialize theta with zeros
        theta = np.zeros(X.shape[1])
        # Train logistic regression for the current class
        theta = gradient_descent(X, y_binary, theta, learning_rate, num_iterations)
        theta_models[i] = theta
    return theta_models


# save theta models to a file as pandas dataframe
def save_theta_models(theta_models, filename):
    df = pd.DataFrame(theta_models)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('datasets/dataset_train.csv')
    #df = pd.read_csv('datasets/train.csv') # to use csv from splitter
    # preprocessing
    X = df.iloc[:, 6:18].values
    y = df.iloc[:, 1].values
    y = np.array([0 if label == 'Ravenclaw' 
                  else 1 if label == 'Slytherin' 
                  else 2 if label == 'Gryffindor' 
                  else 3 for label in y]) # Huffelpuff
    X = np.nan_to_num(X)
    scaler = MinMaxScaler(copy=False)
    X = scaler.fit_transform(X)
    # Hyperparameters
    learning_rate = 1
    num_iterations = 1000
    num_classes = 4
    # Train the models
    theta_models = train_logistic_regression(X, y, num_classes, learning_rate, num_iterations)
    # Save the models' weights
    save_theta_models(theta_models, 'theta.csv')
