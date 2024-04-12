import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    try:
        # Load the dataset and select numerical features
        df = pd.read_csv('datasets/dataset_train.csv')
        df = df.drop(columns=['Index'])
        # Plot the pair plot
        pairplot = sns.pairplot(df, hue='Hogwarts House', diag_kind='hist')
        pairplot.set(xticklabels=[], yticklabels=[])
        # Change the size of the labels
        for ax in pairplot.axes.flatten():
             # Change the size of the x-axis labels
            xlabel = ax.get_xlabel()
            ax.set_xlabel(xlabel, fontsize=8.5)
            # Change the size of the y-axis labels
            ylabel = ax.get_ylabel()
            ax.set_ylabel(ylabel, fontsize=5)
        # Change the size of the title
        plt.setp(pairplot._legend.get_texts(), fontsize='8')
        plt.setp(pairplot._legend.get_title(), fontsize='8')
        plt.show()
    except FileNotFoundError:
        print("Error: Invalid file.")
        exit(1)
