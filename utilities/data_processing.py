# Contains utility functions for data access

# Import statements
import pandas as pd

# returns a usable format of the input data.  Input (str): path to fil to be loaded, type (str): type of file input
# currently only accepts *.csv, can be extended to other data formats if required later in development
def get_data(path, type = 'csv'):
    if type == 'csv':
        return pd.read_csv(path)
    else:
        print('type must be \'csv\'')
