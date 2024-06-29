import pandas as pd
import numpy as np


# Loading data
df = pd.read_csv('dataset-a.csv', on_bad_lines='skip', delimiter="|",)
print('Dataset:\n', df)

# function replace symbol
def replace_symbols(text, replace_dict):
    for symbol, replacement in replace_dict.items():
        text = text.replace(symbol, replacement)
    return text