import numpy as np
import pandas as pd
from logreg_train import sigmoid

if __name__ == "__main__":
    # Load test data
    data = pd.read_csv('datasets/dataset_train.csv')
     # Extract X and y
    X = data.iloc[:, 6:18].values
    # fill null with mean
    X = np.nan_to_num(X)
    # normalize the data
    X /= 500
    

    # Load the first column of thetas from the file into a numpy array
    theta1 = pd.read_csv('theta.csv').values[:, 0]
    theta2 = pd.read_csv('theta.csv').values[:, 1]
    theta3 = pd.read_csv('theta.csv').values[:, 2]
    theta4 = pd.read_csv('theta.csv').values[:, 3]

    # for each row in X_test, calculate the probability of each class
    probabilities = np.array([sigmoid(X @ theta) for theta in [theta1, theta2, theta3, theta4]]).T

    # print the probabilities 
    print(probabilities)
    
    # for each row in X_test, predict the class with the highest probability
    predictions = np.argmax(probabilities, axis=1)

    # Save the predictions to a file in the format index, prediction
    pd.DataFrame({'Index': range(len(predictions)), 'Hogwarts House': predictions}).to_csv('houses.csv', index=False)
    # print('Predictions saved to houses.csv')
