import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def histogram(data, course_cols, house_cols):
    num_features = len(course_cols)
    num_categories = len(data[house_cols].unique())

    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(15, 10 * num_features))

    for i, feature_column in enumerate(course_cols):
        ax = axes[i]
        for category_value in data[house_cols].unique():
            subset = data[data[house_cols] == category_value]
            ax.hist(subset[feature_column], bins=20, alpha=0.5, label=f'Category {category_value}')

        # ax.set_title(f'{feature_column}')
        ax.set_xlabel(feature_column)
        ax.set_ylabel('score')
        # ax.legend()

    # plt.tight_layout()
    plt.show()


def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        course_cols = data.select_dtypes(include=[np.number]).columns
        course_cols = course_cols[1:]

        house_column = data.columns[1]
        histogram(data, course_cols, house_column)
    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)


if __name__ == '__main__':
    main()