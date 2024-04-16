import numpy as np
import pandas as pd
from logreg_train import sigmoid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from test import sklearn_test

def house_from_index(index):
    if index == 0:
        return "Ravenclaw"
    elif index == 1:
        return "Slytherin"
    elif index == 2:
        return "Gryffindor"
    elif index == 3:
        return "Hufflepuff"


if __name__ == "__main__":
    # Load test data
    data = pd.read_csv('datasets/dataset_test.csv')
    #data = pd.read_csv('datasets/test.csv') # to use csv from splitter
    # preprocessing
    X = data.iloc[:, 6:18].values
    X = np.nan_to_num(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y_test = data.iloc[:, 1].values
    y_test = np.array([0 if label == 'Ravenclaw' 
                       else 1 if label == 'Slytherin' 
                       else 2 if label == 'Gryffindor' 
                       else 3 for label in y_test]) # Hufflepuff

    # Load the weights
    theta1 = pd.read_csv('theta.csv').values[:, 0]
    theta2 = pd.read_csv('theta.csv').values[:, 1]
    theta3 = pd.read_csv('theta.csv').values[:, 2]
    theta4 = pd.read_csv('theta.csv').values[:, 3]

    # for each row in X_test, calculate the probability of each class
    probabilities = np.array([sigmoid(X @ theta) for theta in [theta1, theta2, theta3, theta4]]).T

    # for each row in X_test, predict the class with the highest probability
    predictions = np.argmax(probabilities, axis=1)
    predictions = np.array([house_from_index(index) for index in predictions])

    # Save the predictions to a file in the format index, prediction
    pd.DataFrame({'Index': range(len(predictions)), 'Hogwarts House': predictions}).to_csv('houses.csv', index=False)

    
    # Compare with sklearn
    sklearn_test('datasets/dataset_test.csv')
    #sklearn_test('datasets/test.csv') # to use csv from splitter
    sklearn_predictions = pd.read_csv('houses2.csv').iloc[:, 1].values

    # Print the accuracy of the model
    print("Accuracy:", accuracy_score(predictions, sklearn_predictions))
