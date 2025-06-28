################################################################################
# Hand-On Data Preprocessing in Python - Chapter 001
################################################################################

# Index
#
# Overview of the basic functions of NumPy
#   Basic statistics using NumPy
#   Creating NumPy arrays
#   The np.arange() function
#   The np.zeros() function
#   The np.ones() function
#   The np.linspace() function
#   Slicing
#
# Overview of the basic functions of Pandas
#   Read CSV
#   Explore dataset
#   Pandas data access - access rows
#   Pandas data access - access columns
#   Pandas data access - access values (from a Series - row)
#   Pandas data access - access values (from a Series - column)
#   Slicing
#   Boolean masking for filtering a DataFrame
#   Pandas functions for exploring a DataFrame
#   Pandas functions for exploring a DataFrame - numerical columns
#   Pandas functions for exploring a DataFrame - categorical columns
#   Pandas applying a function
#   Pandas applying a Lambda function
#   Pandas applying a function to a DataFrame
#   The Pandas groupby function
#   Pandas multi-level indexing - The .unstack() function
#   Pandas multi-level indexing - The .stack() function
#   Pandas .pivot() and .melt() functions

# Initial configuration
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_adult_csv = os.path.join(script_dir, "../data/adult/adult.csv")
file_path_wide_csv = os.path.join(script_dir, "../data/wide/wide.csv")
file_path_long_csv = os.path.join(script_dir, "../data/long/long.csv")

################################################################################
# Overview of the basic functions of NumPy
################################################################################

import numpy as np

# Test object
list_nums = [1, 2, 3, 4, 5, 6]

# Basic statistics using NumPy
mean = np.mean(list_nums)
median = np.median(list_nums)
std_dev = np.std(list_nums)
variance = np.var(list_nums)
minimum = np.min(list_nums)
maximum = np.max(list_nums)
sum_of_list = np.sum(list_nums)
count = len(list_nums)
unique_values = np.unique(list_nums)
frequency = {value: list_nums.count(value) for value in unique_values}

# Creating NumPy arrays
array = np.array(list_nums)

# The np.arange() function
range_0_to_6 = np.arange(6)
range_6_to_16 = np.arange(6, 16)
range_sub_6_to_16_step_0_6 = np.arange(-6, 16, 0.6)

# The np.zeros() and np.ones() functions
array_6x1_zeros = np.zeros(6)
array_6x6_zeros = np.zeros([6, 6])
array_6x1_ones = np.ones(6)
array_6x6_ones = np.ones([6, 6])

# The np.linspace() function
linespace = np.linspace(0, 10, 6)

# Slicing
my_array = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]])
my_array[1, 1]
my_array[1, :]
my_array[:, 1]
my_array[1:3, :]
my_array[1:3, 0:2]
my_array[1:3, [0,2]]

################################################################################
# Overview of the basic functions of Pandas
################################################################################
import pandas as pd

# Read CSV
adult_df = pd.read_csv(file_path_adult_csv)

# Explore dataset
first_5_rows = adult_df.head()
last_5_rows = adult_df.tail()
columns_names = adult_df.loc[0].index
columns_names = adult_df.columns
columns_index = adult_df.age.index      # start=0, stop=32561, step=1

# Pandas data access - access rows
change_index = adult_df.set_index(np.arange(10000, 42561), inplace=True)
second_row = adult_df.loc[10001]
second_row = adult_df.iloc[1]

# Pandas data access - access columns
edu_column = adult_df.education
edu_column = adult_df['education']

# Pandas data access - access values (from a Series - row)
row_series = adult_df.loc[10002]
same_data = row_series.loc['education']
same_data = row_series.iloc[3]
same_data = row_series['education']
same_data = row_series.education

# Pandas data access - access values (from a Series - column)
column_series = adult_df.education
same_data = column_series.loc[10002]
same_data = column_series.iloc[2]
same_data = column_series.loc[10002]

# Slicing
adult_df.loc[:, 'education':'occupation']
adult_df.sort_values('education-num').reset_index().iloc[1:32561:3617]

# Boolean masking for filtering a DataFrame
twopowers_sr = pd.Series([2**i for i in range(11)])
BM = [False, False, False, True, False, False, False, True, True, True, True]
x = twopowers_sr[BM]
BM = twopowers_sr < 500
BM = twopowers_sr >= 500
x = twopowers_sr[BM]
twopowers_sr[twopowers_sr >= 500]
BM = adult_df.education == 'Preschool'
BM = adult_df['education-num'] <= 1
x = np.mean(adult_df[BM].age)
y = np.median(adult_df[BM].age)
BM1 = adult_df['education-num'] > 10
BM2 = adult_df['education-num'] < 10
adult_df[BM1].shape[0]
adult_df[BM2].shape[0]
np.mean(adult_df[BM1].capitalGain)
np.mean(adult_df[BM2].capitalGain)

# Pandas functions for exploring a DataFrame
adult_df.shape
adult_df.shape[0]
adult_df.shape[1]
adult_df.columns
adult_df.info()

# Pandas functions for exploring a DataFrame - numerical columns
adult_df.describe()
adult_df.age.plot.hist()
adult_df.age.plot.hist(bins=18, edgecolor='black', title='Age Distribution', xlabel='Age', ylabel='Frequency')
adult_df['age'].plot.box()
adult_df['age'].plot.box(vert=False, title='Age Boxplot', ylabel='Age')
adult_df.relationship.value_counts().plot.bar()

# Pandas functions for exploring a DataFrame - categorical columns
adult_df.relationship.unique()
adult_df.relationship.value_counts()
adult_df.relationship.value_counts(normalize=True)

# Pandas applying a function
def generic_function(n):
    return n * 2
adult_df.age.apply(generic_function)
total_fnlwgt = adult_df.fnlwgt.sum()
def calculate_percentage(n):
    return n / total_fnlwgt * 100
adult_df.fnlwgt = adult_df.fnlwgt.apply(calculate_percentage)

# Pandas applying a Lambda function
total_fnlwgt = adult_df.fnlwgt.sum()
adult_df.fnlwgt = adult_df.fnlwgt.apply(lambda v: v / total_fnlwgt * 100)

# Pandas applying a function to a DataFrame
def calculate(row):
    return row.age - row['education-num']
adult_df.apply(calculate, axis=1)
adult_df.apply(lambda r: r.age - r['education-num'], axis=1)
adult_df['lifeNoEd'] = adult_df.apply(lambda r: r['age'] - r['education-num'], axis=1)
adult_df['capitalNet'] = adult_df.apply(lambda r: r.capitalGain - r.capitalLoss, axis=1)
adult_df[['education-num', 'lifeNoEd', 'capitalNet']].corr()

# The Pandas groupby function
adult_df.groupby('marital-status').size()
adult_df.groupby(['marital-status', 'sex']).age.median()
adult_df.groupby(['race', 'sex']).capitalNet.mean()

# Pandas multi-level indexing - The .unstack() function
grb_result = adult_df.groupby(['race', 'sex']).capitalNet.mean()
grb_result.unstack()
mlt_series = adult_df.groupby(['race', 'sex', 'income']).fnlwgt.mean()
mlt_series.unstack()
mlt_series.unstack().unstack()

# Pandas multi-level indexing - The .stack() function
mlt_df = mlt_series.unstack().unstack()
mlt_df.stack()
mlt_df.stack().stack()

# Pandas .pivot() and .melt() functions
wide_df = pd.read_csv(file_path_wide_csv)
wide_df.melt(
    id_vars="ReadingDateTime",
    value_vars=["NO", "NO2", "NOX", "PM10", "PM2.5"],
    var_name="Species",
    value_name="Value"
)
long_df = pd.read_csv(file_path_long_csv)
long_df.pivot(
    index="ReadingDateTime",
    columns="Species",
    values="Value"
)