import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    try:
        df = pd.read_csv('datasets/dataset_train.csv')
        df = df.drop(columns=['Index'])
        pairplot = sns.pairplot(df, hue='Hogwarts House', diag_kind='hist')
        pairplot.set(xticklabels=[], yticklabels=[])
        for ax in pairplot.axes.flatten():
            ax.tick_params(axis='both', which='major', labelsize=1)
        plt.setp(pairplot._legend.get_texts(), fontsize='6')
        plt.setp(pairplot._legend.get_title(), fontsize='8')
        plt.show()
    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)
