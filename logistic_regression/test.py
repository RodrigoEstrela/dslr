from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def house_from_index(index):
    if index == 0:
        return "Ravenclaw"
    elif index == 1:
        return "Slytherin"
    elif index == 2:
        return "Gryffindor"
    elif index == 3:
        return "Hufflepuff"

def sklearn_test(data_path):
    # Load training data
    train_data = pd.read_csv('datasets/dataset_train.csv')
    #train_data = pd.read_csv('datasets/train.csv') # to use csv from splitter
    X_train = train_data.iloc[:, 6:18].values
    X_train = np.nan_to_num(X_train)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = train_data.iloc[:, 1].values
    y_train = np.array([0 if label == 'Ravenclaw' 
                        else 1 if label == 'Slytherin' 
                        else 2 if label == 'Gryffindor' 
                        else 3 for label in y_train]) # Hufflepuff

    # Load test data
    test_data = pd.read_csv(data_path)
    X_test = test_data.iloc[:, 6:18].values
    X_test = np.nan_to_num(X_test)
    X_test = scaler.fit_transform(X_test)
    y_test = test_data.iloc[:, 1].values
    y_test = np.array([0 if label == 'Ravenclaw' 
                       else 1 if label == 'Slytherin' 
                       else 2 if label == 'Gryffindor' 
                       else 3 for label in y_test]) # Hufflepuff

    # Create and train the model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    y_pred = np.array([house_from_index(index) for index in y_pred])

    # Create a DataFrame with the predictions
    results = pd.DataFrame({
        'Index': range(len(y_pred)),
        'Hogwarts House': y_pred
    })

    # Write the DataFrame to a CSV file
    results.to_csv('houses2.csv', index=False)

if __name__ == '__main__':
    sklearn_test('datasets/dataset_train.csv')
    #sklearn_test('datasets/train.csv') # to use csv from splitter
