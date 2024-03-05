import numpy as np
import pandas as pd
import sys

def std(column):
    mean = sum(column) / len(column)
    return (sum((column - mean) ** 2) / len(column)) ** 0.5

def min(column):
    min = column[0]
    for i in column:
        if i < min:
            min = i
    return min

def quantile(column, q):
    column = sorted(column)
    index = (len(column) - 1) * q
    if index.is_integer():
        return column[int(index)]
    else:
        return column[int(index)] + (column[int(index) + 1] - column[int(index)]) * (index - int(index))
    
def max(column):
    max = column[0]
    for i in column:
        if i > max:
            max = i
    return max


def ft_describe(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols[1:]
    result = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'])

    for column in numerical_cols:
        data[column] = data[column].fillna(data[column].mean())
        result[column] = [
            data[column].notnull().sum(),
            sum(data[column]) / len(data[column]),
            std(data[column]),
            min(data[column]),
            quantile(data[column], 0.25),
            quantile(data[column], 0.5),
            quantile(data[column], 0.75),
            max(data[column])
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
