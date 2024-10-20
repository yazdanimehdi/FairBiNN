import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def work_func(x):
    if x == 'Private':
        return 0
    elif x == 'State-gov':
        return 1
    elif x == 'Self-emp-not-inc':
        return 2
    elif x == 'Self-emp-inc':
        return 3
    elif x == 'Federal-gov':
        return 4
    elif x == 'Local-gov':
        return 5
    elif x == 'Without-pay':
        return 6


def education_func(x):
    if x == 'Masters':
        return 0
    elif x == '9th':
        return 1
    elif x == 'Some-college':
        return 2
    elif x == 'Assoc-acdm':
        return 3
    elif x == 'HS-grad':
        return 4
    elif x == '11th':
        return 5
    elif x == 'Bachelors':
        return 6
    elif x == '10th':
        return 7
    elif x == 'Assoc-voc':
        return 8
    elif x == '7th-8th':
        return 9
    elif x == '5th-6th':
        return 10
    elif x == '12th':
        return 11
    elif x == 'Doctorate':
        return 12
    elif x == 'Prof-school':
        return 13
    elif x == 'Preschool':
        return 14


def marital_func(x):
    if x == 'Married-civ-spouse':
        return 0
    elif x == 'Never-married':
        return 1
    elif x == 'Widowed':
        return 2
    elif x == 'Divorced':
        return 3
    elif x == 'Separated':
        return 4
    elif x == 'Married-spouse-absent':
        return 5
    elif x == 'Married-AF-spouse':
        return 6


def occupation_func(x):
    if x == 'Sales':
        return 0
    elif x == 'Farming-fishing':
        return 1
    elif x == 'Transport-moving':
        return 2
    elif x == 'Exec-managerial':
        return 3
    elif x == 'Craft-repair':
        return 4
    elif x == 'Prof-specialty':
        return 5
    elif x == 'Other-service':
        return 6
    elif x == 'Tech-support':
        return 7
    elif x == 'Adm-clerical':
        return 8
    elif x == 'Machine-op-inspct':
        return 9
    elif x == 'Handlers-cleaners':
        return 10
    elif x == 'Protective-serv':
        return 11
    elif x == 'Priv-house-serv':
        return 12
    elif x == 'Armed-Forces':
        return 13


def relationship_func(x):
    if x == 'Husband':
        return 0
    elif x == 'Other-relative':
        return 1
    elif x == 'Wife':
        return 2
    elif x == 'Unmarried':
        return 3
    elif x == 'Own-child':
        return 4
    elif x == 'Not-in-family':
        return 5


def race_func(x):
    if x == 'White':
        return 0
    elif x == 'Black':
        return 1
    elif x == 'Other':
        return 2
    elif x == 'Asian-Pac-Islander':
        return 3
    elif x == 'Amer-Indian-Eskimo':
        return 4


def sex_func(x):
    if x == 'Male':
        return 0
    elif x == 'Female':
        return 1


def country_func(x):
    if x == 'France':
        return 0
    elif x == 'United-States':
        return 1
    elif x == 'Germany':
        return 2
    elif x == 'Mexico':
        return 3
    elif x == 'Philippines':
        return 4
    elif x == 'Poland':
        return 5
    elif x == 'Cuba':
        return 6
    elif x == 'El-Salvador':
        return 7
    elif x == 'India':
        return 8
    elif x == 'Puerto-Rico':
        return 9
    elif x == 'Canada':
        return 10
    elif x == 'Thailand':
        return 11
    elif x == 'Vietnam':
        return 12
    elif x == 'England':
        return 13
    elif x == 'Haiti':
        return 14
    elif x == 'Italy':
        return 15
    elif x == 'Greece':
        return 16
    elif x == 'Outlying-US(Guam-USVI-etc)':
        return 17
    elif x == 'Japan':
        return 18
    elif x == 'Yugoslavia':
        return 19
    elif x == 'China':
        return 20
    elif x == 'Guatemala':
        return 21
    elif x == 'Honduras':
        return 22
    elif x == 'Jamaica':
        return 23
    elif x == 'Peru':
        return 24
    elif x == 'Dominican-Republic':
        return 25
    elif x == 'Ireland':
        return 26
    elif x == 'Portugal':
        return 27
    elif x == 'Taiwan':
        return 28
    elif x == 'Iran':
        return 29
    elif x == 'South':
        return 30
    elif x == 'Hong':
        return 31
    elif x == 'Ecuador':
        return 32
    elif x == 'Nicaragua':
        return 33
    elif x == 'Laos':
        return 34
    elif x == 'Cambodia':
        return 35
    elif x == 'Columbia':
        return 136
    elif x == 'Scotland':
        return 37
    elif x == 'Trinadad&Tobago':
        return 38
    elif x == 'Hungary':
        return 39
    elif x == 'Holand-Netherlands':
        return 40


def y_func(x):
    if '<=50K' in x:
        return 0
    elif '>50K' in x:
        return 1


def pre_process_and_load_adult(mode="cardinal", train=True):
    scaler = MinMaxScaler()

    if train:
        df_raw = pd.read_csv('./data/adult.data', sep=', ', engine='python')
    else:
        df_raw = pd.read_csv('./data/adult.test', sep=', ', engine='python')
    if mode == "onehot":
        df_raw['Y'] = df_raw['Y'].apply(y_func)
        df_raw = df_raw.drop('fnlwgt', axis=1)
        df_raw.replace('?', np.NAN, inplace=True)
        df_raw['sex'] = df_raw['sex'].apply(sex_func)
        df_raw.dropna(inplace=True, axis=0)
        df_train_processed = pd.DataFrame()
        for column in df_raw.columns:
            if column not in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                              'native-country']:
                df_train_processed[column] = df_raw[column]
            else:
                dummies = pd.get_dummies(df_raw[column], prefix=column, dtype=float)
                df_train_processed = pd.concat([df_train_processed, dummies], axis=1)

        df_raw = df_train_processed
    else:
        df_raw['workclass'] = df_raw['workclass'].apply(work_func)
        df_raw['education'] = df_raw['education'].apply(education_func)
        df_raw['marital-status'] = df_raw['marital-status'].apply(marital_func)
        df_raw['occupation'] = df_raw['occupation'].apply(occupation_func)
        df_raw['relationship'] = df_raw['relationship'].apply(relationship_func)
        df_raw['race'] = df_raw['race'].apply(race_func)
        df_raw['sex'] = df_raw['sex'].apply(sex_func)
        df_raw['native-country'] = df_raw['native-country'].apply(country_func)
        df_raw['Y'] = df_raw['Y'].apply(y_func)
        df_raw.dropna(inplace=True)

    if "native-country_Holand-Netherlands" in df_raw.columns:
        df_raw = df_raw.drop(['native-country_Holand-Netherlands'], axis=1)

    y = df_raw['Y']
    X = df_raw.drop(['Y'], axis=1)
    X = scaler.fit_transform(X)
    X = np.array(X)
    y = np.array(y)
    s = np.array(df_raw['sex'])

    return X, y, s