import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold

df = pd.read_csv(
    'gender_classification.csv')

df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
scaler = preprocessing.MinMaxScaler()
df[['forehead_height_cm', 'forehead_width_cm']] = scaler.fit_transform(
    df[['forehead_height_cm', 'forehead_width_cm']])

df_shuffled = df.sample(frac=1).reset_index(drop=True)

dfX = df_shuffled.iloc[:, :-1]
dfY = df_shuffled.iloc[:, -1]

# K Frost Validation
k = 10
kf = KFold(n_splits=10)
i = 0
print(df_shuffled.head())
for train, test in kf.split(dfX):
    #print("%s %s" % (train, test))
    X_train, X_test = dfX.iloc[train], dfX.iloc[test]
    y_train, y_test = dfY.iloc[train], dfY.iloc[test]

    X_train.to_csv('train/trainX' +
                   str(i) + '.pts', header=None, index=None, sep='	', mode='a')
    X_test.to_csv('test/testX' +
                  str(i) + '.pts', header=None, index=None, sep='	', mode='a')
    y_train.to_csv('train/trainY' +
                   str(i) + '.pts', header=None, index=None, sep='	', mode='a')
    y_test.to_csv('test/testY' +
                  str(i) + '.pts', header=None, index=None, sep='	', mode='a')
    i += 1
