import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def calculate_similarity(feature1, feature2):
    correlation = df[feature1].corr(df[feature2])
    return correlation

def find_most_similar_pair():
    features = df.columns
    max_similarity = -1
    most_similar_pair = None
    counter = 0
    reducer = 0
    for feature1, feature2 in combinations(features, 2):
        similarity = calculate_similarity(feature1, feature2)
        print(f"{feature1} and {feature2}: {format(similarity, '.4f')}")
        counter += 1
        if counter == 12 - reducer:
            print("...")
            counter = 0
            reducer += 1
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_pair = (feature1, feature2)

    return most_similar_pair

if __name__ == "__main__":
    try:
        df = pd.read_csv('datasets/dataset_train.csv')
        df = df.drop(columns=['Index'])
        df = df.select_dtypes(include=[float, int])
        # fill missing values with mean
        # for column in df.columns:
        #     df[column] = df[column].fillna(df[column].mean())

        most_similar_pair = find_most_similar_pair()
        
        if most_similar_pair:
            feature1, feature2 = most_similar_pair

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature1], y=df[feature2])
            plt.title(f'Scatter Plot of {feature1} vs {feature2}')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.show()

            print(f"The most similar features are: {feature1} and {feature2}")
        else:
            print("No similar pair found.")
    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)
