import pandas as pd
from validation import check_validity


if __name__ == '__main__':
    preds = pd.read_csv('valid.csv')
    binary, score = check_validity(preds)
    print(binary)