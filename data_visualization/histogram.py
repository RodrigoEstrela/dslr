import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def histogram(data, course_cols, house_cols):
    num_features = len(course_cols)
    categories = data[house_cols].unique()


    _, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 3 * num_features))

    counter = 0
    for i, feature_column in enumerate(course_cols):
        ax = axes[i]
        for category_value in categories:
            subset = data[data[house_cols] == category_value]
            ax.hist(subset[feature_column], bins=20, alpha=0.5, label=f'House {category_value}')

        ax.set_xlabel(feature_column)
        ax.set_xticks([])
        if counter == 0:
            ax.legend()
            counter += 1
        
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def main():
    try:
        # Load the dataset and select numerical features
        data = pd.read_csv("datasets/dataset_train.csv")
        course_cols = data.select_dtypes(include=[np.number]).columns
        course_cols = course_cols[1:]
        # Plot the histogram
        house_column = data.columns[1]
        histogram(data, course_cols, house_column)
    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)


if __name__ == '__main__':
    main()
