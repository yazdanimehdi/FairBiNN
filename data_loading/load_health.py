import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def gather_labels(df):
    labels = []
    for j in range(df.shape[1]):
        if type(df[0, j]) is str:
            labels.append(np.unique(df[:, j]).tolist())
        else:
            labels.append(np.median(df[:, j]))
    return labels


def pre_process_and_load_health():
    raw_df = pd.read_csv("./data/health.csv")
    raw_df = raw_df[raw_df['YEAR_t'] == 'Y3']
    sex = raw_df['sexMISS'] == 0
    age = raw_df['age_MISS'] == 0
    raw_df = raw_df.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    raw_df = raw_df[sex & age]
    ages = raw_df[[f'age_{i}5' for i in range(0, 9)]]
    sexs = raw_df[['sexMALE', 'sexFEMALE']]
    charlson = raw_df['CharlsonIndexI_max']

    x = raw_df.drop(
        [f'age_{i}5' for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max',
                                              'CharlsonIndexI_min',
                                              'CharlsonIndexI_ave', 'CharlsonIndexI_range',
                                              'CharlsonIndexI_stdev',
                                              'trainset'], axis=1)
    u = ages.to_numpy().argmax(axis=1)
    x['age'] = u
    x['sex'] = sexs.to_numpy().argmax(axis=1)
    y = (charlson.to_numpy() > 0).astype(np.float32)
    X, X_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    u = np.array(X['age'])
    u_test = np.array(X_test['age'])
    X = np.array(X)
    X_test = np.array(X_test)
    y = np.array(y)
    y_test = np.array(y_test)

    return X, y, u, X_test, y_test, u_test