import numpy as np
import pandas as pd
import sys


def ft_describe(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols[1:]
    result = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

    for column in numerical_cols:
        data[column] = data[column].fillna(data[column].mean())
        result[column] = [
            len(data[column]),
            data[column].mean(),
            data[column].std(),
            data[column].min(),
            data[column].quantile(0.25),
            data[column].median(),
            data[column].quantile(0.75),
            data[column].max()
        ]
    return result


def main():
    try:
        # load dataset
        data = pd.read_csv(sys.argv[1])
        # print ft_describe
        print(ft_describe(data))

    except:
        print("Error: Invalid file.")
        exit(1)


if __name__ == '__main__':
    main()
