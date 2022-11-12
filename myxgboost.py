import numpy as np
import pandas as pd
from datetime import date
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, ensemble

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

def mypredict(train, test, next_fold, t):
    
    if t!=1:
        train = pd.concat([train,next_fold],ignore_index=True)

    # not all depts need prediction
    
    start_date = pd.to_datetime("2011-03-01") + relativedelta(months=2 * (t-1))
    end_date = pd.to_datetime("2011-05-01") + relativedelta(months=2 * (t-1))

    # find_week = lambda x : x.isocalendar()[1]+1  if x.isocalendar()[0] == 2010 else x.isocalendar()[1]

    find_week = lambda x : x.isocalendar()[1]
    find_yr = lambda x : x.isocalendar()[0]

    test['Wk'] =  pd.to_datetime(test['Date']).apply(find_week)
    train['Wk'] =  pd.to_datetime(train['Date']).apply(find_week)

    test['Yr'] =  pd.to_datetime(test['Date']).apply(find_yr)
    train['Yr'] =  pd.to_datetime(train['Date']).apply(find_yr)

    time_ids = (pd.to_datetime(test['Date'])>=start_date)&(pd.to_datetime(test['Date'])<end_date)
    test_current = test.loc[time_ids,]

    test_depts = test_current.Dept.unique()
    test_pred = None
    
    for dept in test_depts:    
    # no need to consider stores that do not need prediction
    # or do not have training samples
        #print(dept)
        train_dept_data = train[train['Dept']==dept]
        test_dept_data = test_current[test_current['Dept']==dept]
        train_stores = train_dept_data.Store.unique()
        test_stores = test_dept_data.Store.unique()
        test_stores = np.intersect1d(train_stores, test_stores)
        
        if len(test_stores) == 0:
            continue
        
        train_y_before_svd = []
        train_dept_data["time"] = train_dept_data.Wk.astype("str") + train_dept_data.Yr.astype("str")
        all_wk = pd.DataFrame(train_dept_data.time.unique(),columns = ["time"])
        
        for store in test_stores:
            
            
            tmp_train = train_dept_data[train_dept_data['Store']==store]
            tmp_train["time"] = tmp_train.Wk.astype("str") + tmp_train.Yr.astype("str")
            tmp_data = pd.merge(all_wk,tmp_train,on="time",how="left")
            

            trainY = tmp_data['Weekly_Sales'].fillna(1e-7).values
            train_y_before_svd.append(trainY)
        
        train_y_before_svd = np.array(train_y_before_svd)
        #print(train_y_before_svd.shape)
        train_y_before_svd_mean = train_y_before_svd.mean(axis = 1)
        train_y_mean = train_y_before_svd - train_y_before_svd_mean.reshape(-1,1)
        u,sigma,vt = np.linalg.svd(train_y_mean)
        d = min(10,train_y_mean.shape[0])
        u = u[:,:d]
        sigma = sigma[:d]
        vt = vt[:d,:]
        train_y = u@np.diag(sigma)@vt + train_y_before_svd_mean.reshape(-1,1)
        
        train_y_dict = {}
        for i in range(len(test_stores)):
            trainY = train_y[i]
            trainY = trainY[train_y_before_svd[i]!=1e-7]
            train_y_dict[test_stores[i]] = trainY
            
    
        for store in test_stores:
            #print(store)
            tmp_train = train_dept_data[train_dept_data['Store']==store]
            tmp_test = test_dept_data[test_dept_data['Store']==store]


            trainY = train_y_dict[store]
            tmp_train = tmp_train.drop(['Weekly_Sales'],axis=1)

            ohe = OneHotEncoder(handle_unknown='ignore',sparse=False,drop='if_binary')
            enc = ohe.fit_transform(tmp_train[['Wk','IsHoliday','Yr']])

            ### encoding features ###
            train_dummy = pd.DataFrame(enc,columns=ohe.get_feature_names_out())
            test_dummy = pd.DataFrame(ohe.transform(tmp_test[['Wk','IsHoliday','Yr']]),columns=ohe.get_feature_names_out())

            ### covert to dataframe ###
            train_dummy['Yr'] = tmp_train['Yr'].to_numpy()
            train_dummy['Store'] = tmp_train['Store'].to_numpy()
            train_dummy['Dept'] = tmp_train['Dept'].to_numpy()

            test_dummy['Yr'] = tmp_test['Yr'].to_numpy()
            test_dummy['Store'] = tmp_test['Store'].to_numpy()
            test_dummy['Dept'] = tmp_test['Dept'].to_numpy()

            new_col_list = ohe.get_feature_names_out()
            # print(f'new_col_list:{new_col_list}')
            new_col_list = new_col_list.tolist() + ['Dept','Store']
            # print(f'new_col_list:{new_col_list}')
            train_dummy = train_dummy[new_col_list]
            test_dummy = test_dummy[new_col_list]

            ### to do SVD or PCA ###
            
            
            params = {"n_estimators": 500,
                      "max_depth": 4,
                      "min_samples_split": 5,
                      "learning_rate": 0.01,
                      "loss": "squared_error",
            }



            reg = XGBRegressor(learning_rate = 0.3, 
                               max_depth = 6, 
                               n_estimators = 50,
                               random_state = 4777,
                               verbosity = 0)
            reg.fit(train_dummy,trainY)
            # reg = Ridge(alpha = 0.15)

            #reg = ensemble.GradientBoostingRegressor(**params)
            #print(train_dummy.shape,trainY.shape)
            # reg.fit(train_dummy, trainY)
            # mycoef = reg.coef_
            # myintercept = reg.intercept_
            # mycoef[np.isnan(mycoef)] = 0

            # mycoef[np.abs(mycoef)>10e8] = 0

            # if myintercept == np.nan:
            #     myintercept = 0

            # print(f'mycoef:{mycoef}')
            # exit()

            # tmp_pred = myintercept + np.dot(test_dummy,mycoef).reshape(-1,1)

            
            tmp_pred = reg.predict(test_dummy)
            tmp_test['Weekly_Pred'] = tmp_pred
            tmp_test = tmp_test.drop(['Wk','Yr'],axis=1)

            test_pred = pd.concat([test_pred, tmp_test])
            # exit()

    # print(test_pred.shape)

    
    return train,test_pred



if __name__ == '__main__':

    train = pd.read_csv('train_ini.csv', parse_dates=['Date'])
    test = pd.read_csv('test.csv', parse_dates=['Date'])

    # save weighed mean absolute error WMAE
    n_folds = 10
    next_fold = None
    wae = []

    for t in range(1, n_folds+1):
        print(f'Fold{t}...')

        # *** THIS IS YOUR PREDICTION FUNCTION ***
        train, test_pred = mypredict(train, test, next_fold, t)

        # Load fold file
        # You should add this to your training data in the next call to mypredict()
        fold_file = 'fold_{t}.csv'.format(t=t)
        next_fold = pd.read_csv(fold_file, parse_dates=['Date'])

        # extract predictions matching up to the current fold
        scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

        # extract weights and convert to numpy arrays for wae calculation
        weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
        actuals = scoring_df['Weekly_Sales'].to_numpy()
        preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

        wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())
        print((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())
        # print('WAE:',wae)
        # print(sum(wae)/len(wae))
        # exit()
    # t = 1
    # print(f'Fold{t}...')

    # # *** THIS IS YOUR PREDICTION FUNCTION ***
    # train, test_pred = mypredict(train, test, next_fold, t)

    # # Load fold file
    # # You should add this to your training data in the next call to mypredict()
    # fold_file = 'fold_{t}.csv'.format(t=t)
    # next_fold = pd.read_csv(fold_file, parse_dates=['Date'])

    # # extract predictions matching up to the current fold
    # scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

    # # extract weights and convert to numpy arrays for wae calculation
    # weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
    # actuals = scoring_df['Weekly_Sales'].to_numpy()
    # preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

    # wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

    print('WAE:',wae)
    print(sum(wae)/len(wae))