import pandas as pd
from datetime import timedelta
from joblib import Parallel, delayed
import multiprocessing
import time

# Import original Data
data = pd.read_csv('~/PycharmProjects/HunterDouglas/QualityData.csv',sep=',', encoding='latin-1', error_bad_lines=False, dtype = {'ORIGINAL_ORDER':str}, parse_dates = ['SO_CREATED_DATE'])
data = data.sort_values(['ORIGINAL_ORDER'])

### Remove rows where order is NA
df = data[pd.notnull(data['ORIGINAL_ORDER'])]

## remove rows where original_order_line = 99
df = df[df['ORIGINAL_ORDER_LINE'] != 99]

## remove rows where net sales unit is negative
df = df[df['NET_SALES_UNITS'] >= 0]

## Keep only ORDER_REASON_ID = STD, CON, REM, REP
df = df[df['ORDER_REASON_ID'].isin(['STD','CON','REM','REP'])]
df['ORDER_REASON_ID'] = df['ORDER_REASON_ID'].replace(['CON', 'STD','REM','REP'],['CONSTD','CONSTD','REMREP','REMREP'])

## Keep only RESPONSIBILITY_CODE_ID = ME, WA, NA
df = df.loc[(df['RESPONSIBILITY_CODE_ID'].isin(['ME','WA'])) | (df['RESPONSIBILITY_CODE_ID'].isna()) ]

## Keep one factory and one product
#df = df.loc[(df['ORIGINAL_PLANT'].isin(['B', 'A'])) & (df['PRODUCT_CATEGORY'].isin(['07 Roller Shades', 'Newstyle Hybrid Shutters'])) ]

#### Sort values by:
df = df.sort_values(['ORIGINAL_ORDER', 'SO_CREATED_DATE'])

### Need to reset index after sorting values
df = df.reset_index(drop = True)


### Pivot Table for ORDER_REASON_ID
df.groupby('ORDER_REASON_ID').apply(lambda x: len(x))


### Filter date using split-apply-combine (pandas)
def filter_date(temp):
    temp = temp.reset_index(drop = True)
    temp = temp[temp['SO_CREATED_DATE'] <= temp.loc[0,'SO_CREATED_DATE'] + timedelta(days=90) ]
    return temp


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)



### Filter orders greater than 90 days

## Normal Process
'''
start = time.time()
data_final = df.groupby('ORIGINAL_ORDER').apply(lambda x: filter_date(x))
end = time.time()
print("Execution time normal :" + str(end - start))
'''

### Parrallel process
start = time.time()
data_final = applyParallel(df.groupby('ORIGINAL_ORDER'), filter_date)
end = time.time()
print("Execution time normal :" + str(end - start))


### Need to reset index
data_final = data_final.reset_index(drop = True)

### Write data file to disk
data_final.to_csv('data_after_date_filter.csv', header=True)

#### Import clean data and set it as df
df =  pd.read_csv('~/PycharmProjects/HunterDouglas/data_after_date_filter.csv', index_col=0)


### Getting bad orders
order_list_bad = tuple(df['ORIGINAL_ORDER'][df['ORDER_REASON_ID'].isin(['REMREP'])].unique())
df_bad_order = df[df['ORIGINAL_ORDER'].isin(order_list_bad)]
df_bad_order = df_bad_order.sort_values(['ORIGINAL_ORDER', 'SO_CREATED_DATE'])


### Pivot Table for ORDER_REASON_ID
df_bad_order.groupby('ORDER_REASON_ID').apply(lambda x: len(x))


def filter_repeating_rows(test):
    test1 = test.loc[test['ORIGINAL_ORDER'].astype('str') == test['SALES_ORDER'].astype('str')]
    test1 = test1.loc[test1['ORIGINAL_ORDER_LINE'].astype('int') == 0]
    test_list = tuple(test['ORIGINAL_ORDER_LINE'][test['ORDER_REASON_ID'].isin(['REMREP'])].unique())
    rows_removed = test1[test1['SALES_ORDER_LINE'].isin(test_list)].index
    return test.drop(rows_removed)

### Filter repeating rows - normal
#data_final_bad = df_bad_order.groupby('ORIGINAL_ORDER').apply(lambda x: filter_repeating_rows(x))

### Filter repeating rows - Parallel
data_final_bad = applyParallel(df.groupby('ORIGINAL_ORDER'), filter_repeating_rows)


### Need to reset index
data_final_bad = data_final_bad.reset_index(drop = True)

### Pivot Table for ORDER_REASON_ID
print(data_final_bad.groupby('ORDER_REASON_ID').apply(lambda x: len(x)))


### Sampling in order to maintain equal distribution of classes
### This step is not needed if the classes in data_final_bad are already balanced
'''
df_good_order = df[~df['ORIGINAL_ORDER'].isin(order_list_bad)]
df_good_order = df_good_order.sort_values(['ORIGINAL_ORDER', 'SO_CREATED_DATE'])

order_list_good = tuple(df_good_order['ORIGINAL_ORDER'].unique())

grouped_df = df_good_order.groupby('ORIGINAL_ORDER')
'''
df_sample = pd.DataFrame()
#for i in order_list_good[:1]:
#    df_sample_ = grouped_df.get_group(i)
#    df_sample = df_sample.append(df_sample_)

df_sample = pd.concat([df_sample,data_final_bad])

df_sample = df_sample.reset_index(drop=True)

### Pivot Table for ORDER_REASON_ID
print(df_sample.groupby('ORDER_REASON_ID').apply(lambda x: len(x)))


### Write file to disk
df_sample.to_csv('data_sample.csv')


#### Import clean data and set it as df_sample
df_sample =  pd.read_csv('~/PycharmProjects/HunterDouglas/data_sample.csv', index_col=0)


### Use LabelEncoder to change predictors to integer values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_sample_encoded = pd.DataFrame()
encoded = pd.DataFrame()
for i in range(len(df_sample.columns)):
    encoded = pd.DataFrame(le.fit_transform(df_sample.iloc[:,i].astype('str')))
    df_sample_encoded = pd.concat([df_sample_encoded,encoded], axis = 1)

df_sample_encoded.columns = df_sample.columns

### Write file to disk
df_sample_encoded.to_csv('df_sample_encoded.csv')


#### Import clean data and set it as df_sample_encoded
df_sample_encoded =  pd.read_csv('~/PycharmProjects/HunterDouglas/df_sample_encoded.csv', index_col=0)

### Create seperate X and Y
### Scaling of X
from sklearn.preprocessing import StandardScaler

X = df_sample_encoded.loc[:,df_sample_encoded.columns != 'ORDER_REASON_ID']
Y = df_sample_encoded.loc[:,df_sample_encoded.columns == 'ORDER_REASON_ID']
X_scaled = StandardScaler().fit(X).transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)


from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

## select which estimator to use
estimator = LogisticRegression()

### Use RFE for feature selection
'''
selector = RFE(estimator,1)
selector = selector.fit(X, Y)
selector.ranking_
'''
### remove relevant columns
X = X.loc[:,~X.columns.isin(['ORIGINAL_ORDER','RESPONSIBILITY_CODE_ID', 'SALES_ORDER', 'ORIGINAL_ORDER_LINE','SALES_ORDER_LINE', 'REASON_CODE','REASON_CODE_ID', 'SLATE_SIZE'])]


### Create training and testing sets using StratifiedShuffleSplit
SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_indx, test_indx in SSS.split(X,Y):
    X_train, X_test = X.iloc[train_indx], X.iloc[test_indx]
    Y_train, Y_test = Y.iloc[train_indx], Y.iloc[test_indx]

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
Y_train = Y_train.reset_index(drop=True)
Y_test = Y_test.reset_index(drop=True)


### Fitting the model
model = estimator.fit(X_train, Y_train)

### preciting from the fitted model
predicted_values = pd.DataFrame(model.predict(X_test))

### Create model performance reports
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score
confusion_matrix(Y_test,predicted_values)
report = classification_report(Y_test,predicted_values)
print(report)
jaccard_similarity_score(Y_test,predicted_values)
