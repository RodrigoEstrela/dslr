import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    try:
        df = pd.read_csv('datasets/dataset_train.csv')
        df = df.drop(columns=['Index'])
        df = df.select_dtypes(include=[float, int])
        sns.pairplot(df)
        plt.show()
    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)
