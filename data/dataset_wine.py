import pandas as pd

# Load the dataset from UCI
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

# Save the dataset to a CSV file
df.to_csv('data/wine.csv', index=False)