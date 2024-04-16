import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def calculate_similarity(feature1, feature2):
    # Pearson correlation coefficient
    correlation = df[feature1].corr(df[feature2])

    return correlation

def find_most_similar_pair():
    features = df.columns
    max_similarity = -1
    most_similar_pair = None

    for feature1, feature2 in combinations(features, 2):
        similarity = calculate_similarity(feature1, feature2)

        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_pair = (feature1, feature2)

    return most_similar_pair, max_similarity

if __name__ == "__main__":
    try:
        # Load the dataset and select numerical features
        df = pd.read_csv('datasets/dataset_train.csv')
        df = df.drop(columns=['Index'])
        houses = df['Hogwarts House']
        df = df.select_dtypes(include=[float, int])
        # Find the most similar pair of features
        most_similar_pair, similarity = find_most_similar_pair()

        # Plot the scatter plot of the most similar pair
        if most_similar_pair:
            feature1, feature2 = most_similar_pair
            print(f"The most similar features are: {feature1} and {feature2}\n"
                  f"With {similarity:.2f} of similarity.")

            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[feature1], y=df[feature2], hue=houses)
            plt.title(f'Scatter Plot of {feature1} vs {feature2}')
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.show()

        else:
            print("No similar pair found.")

    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)
