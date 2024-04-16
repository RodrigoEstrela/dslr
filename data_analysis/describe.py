import numpy as np
import pandas as pd
import sys


def line_counter(column):
    counter = 0
    for i in column:
        counter += 1
    return counter


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


# Bonus part --------------------------------------------------------
def bonus_fields(data):
    """
    Calculate descriptive statistics for a given column of data.

    Args:
    - data: A list or numpy array containing the data values.

    Returns:
    - range_val: The range of the data.
    - iqr: The interquartile range (IQR) of the data.
    - cv: The coefficient of variation (CV) of the data.
    """
    # Calculate range
    range_val = np.ptp(data)  # ptp stands for "peak to peak"

    # Calculate interquartile range (IQR)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    # Calculate coefficient of variation (CV)
    mean_val = np.mean(data)
    std_dev = np.std(data)
    cv = (std_dev / mean_val) * 100 if mean_val != 0 else np.nan

    return range_val, iqr, cv
# -------------------------------------------------------------------


def ft_describe(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols[1:]
    result = pd.DataFrame(index=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Range', 'IQR', 'CV'])

    for column in numerical_cols:
        data[column] = data[column].fillna(data[column].mean())
        bonus_stats = bonus_fields(data[column])
        result[column] = [
            line_counter(data[column]), # number of lines
            sum(data[column]) / len(data[column]), # mean
            std(data[column]), # standard deviation
            min(data[column]), # minimum value
            quantile(data[column], 0.25), # 25th percentile
            quantile(data[column], 0.5), # 50th percentile 
            quantile(data[column], 0.75), # 75th percentile
            max(data[column]), # maximum value
            bonus_stats[0], # Range
            bonus_stats[1], # Inter Quartile Range
            bonus_stats[2] # Coefficient of Variation (ratio of the std to the mean)
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
