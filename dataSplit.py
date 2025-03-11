import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file, train_file, test_file):
    data = pd.read_csv(input_file)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

split_data('D://Major Project//MajorDTA//Dataset//Davis.csv', 'D://Major Project//MajorDTA//Dataset//train.csv', 'D://Major Project//MajorDTA//Dataset//test.csv')
