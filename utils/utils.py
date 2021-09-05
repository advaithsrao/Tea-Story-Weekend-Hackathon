
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import re
import datetime
from statsmodels.tsa.stattools import adfuller

def preprocess(data,columns):
    for col in columns:
        data[col] = data[col].apply(lambda x:float(x) if not re.match(r'[a-zA-Z]',str(x)) else None)
    return data


def fill_null(data):
    cols = ['Kolkata_Average_Price','Kolkata_Ref_Price',
            'Bangalore_Average_Price', 'Bangalore_Ref_Price',
            'Cochin_Average_Price', 'Cochin_Ref_Price',
            'Darjeeling_Average_Price','Darjeeling_Ref_Price', 
            'Ernakulam_Average_Price','Ernakulam_Ref_Price', 
            'Siliguri_Average_Price', 'Siliguri_Ref_Price',
            'Guwahati_Average_Price', 'Guwahati_Ref_Price']
    data = data.reset_index(drop=True)
    null_containing_ids = []
    for i in cols:
        nulls=data[data[i].isna()][i].index.values.tolist()
        for j in nulls:
            null_containing_ids.append(j)
    elim_rows = list(set([i for i in null_containing_ids if null_containing_ids.count(i) > 1]))
    non_elim_rows = [x for x in data.index.tolist() if x not in elim_rows]
    new_data = data.iloc[elim_rows,:]
    
    data = data.iloc[non_elim_rows,:]
    
    data = data.reset_index(drop=True)
    count = 14
    for i in cols:
        print('column :',i)
        idx = cols.index(i)
        non_idx = [m+1 for m in range(0,14) if m!=idx]
        print('idx :',idx)
        nulls=data[data[i].isna()][i].index.values.tolist()
        for j in nulls:
            data.iloc[j,idx+1] = (count*data.iloc[j,-1])-sum(data.iloc[j,non_idx])
            print('replacing row {} on column {} and value {}'.format(j,cols[idx],data.iloc[j,idx+1]))
    new_data.iloc[:,1:-1] = new_data.iloc[:,1:-1].interpolate(method='linear',axis = 1)
    data = pd.concat([data,new_data])
    return data.dropna()

### gets errors in predictions vs ground-truth
def errors_list(y_test,y_pred):
    errors = []
    for i in range(len(y_test)):
        err = 0
        for j in range(len(y_test[i])):
            if not np.isnan(y_test[i][j]):
                err += abs(y_pred[i][j]-y_test[i][j])
        errors.append(err)
    return errors

def test_stationarity(timeseries):
    '''
    input : time series data
    output : dataframe post process of testing for stationarity
    premise: performs Dickey-Fuller testtests the time series data 
    '''
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#     std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
