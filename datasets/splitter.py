import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def spliter(path):
	# read the dataset
	df = pd.read_csv(path)

	# get the locations
	X = df.iloc[:, 6:18]
	y = df.iloc[:, 1]
	 
	# split the dataset
	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, test_size=0.3, random_state=0)

	print(f'train size {len(X_train)} test size {len(X_test)}')

	pd.concat([pd.concat([y_train, X_train], axis=1)]).to_csv('train.csv')
	pd.concat([pd.concat([y_test, X_test], axis=1)]).to_csv('test.csv')

if __name__ == '__main__':
	spliter(sys.argv[1])
