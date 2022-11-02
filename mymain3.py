import numpy as np
import pandas as pd
from datetime import date
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression



def preprocess(train, test,next_fold, t):
    
    if t == 1:
        start_date = pd.to_datetime("2010-02-01")
        end_date = start_date + relativedelta(months=(13))
        print(f'start_date:{start_date}')
        print(f'end_date:{end_date}')
        time_ids = (pd.to_datetime(test['Date'])>=start_date)&(pd.to_datetime(test['Date'])<start_date)
        test_current = test.loc[time_ids,]
        holiday_ids = test_current['IsHoliday']==False
        test_current = test_current.loc[holiday_ids,]

        start_last_year = min(test_current['Date']) - 375
        end_last_year = max(test_current['Date']) - 350
        print(f'test_current:{test}')

        print(f'train:{train}')
    
    else:
        start_date = pd.to_datetime("2011-03-01") + relativedelta(months=2 * (t-1))
        end_date = pd.to_datetime("2011-05-01") + relativedelta(months=2 * (t-1))
        tmp1 = pd.to_datetime("2010-12-31").isocalendar()[1]
        tmp2 = pd.to_datetime("2011-12-30").isocalendar()[1]

        find_week = lambda x : x.isocalendar()[1]
        test['Wk'] =  pd.to_datetime(test['Date']).apply(find_week)
        train['Wk'] =  pd.to_datetime(train['Date']).apply(find_week)
        
        time_ids = (pd.to_datetime(test['Date'])>=start_date)&(pd.to_datetime(test['Date'])<end_date)
        tmp_time_ids = (pd.to_datetime(test['Date'])>=start_date)
        test_current = test.loc[time_ids,]
        # print(f'test_current:{test_current}')
        holiday_ids = test_current['IsHoliday']==False
        test_current = test_current.loc[holiday_ids,]

        
        # print(f'test_current:{test_current}')
        start_last_year = test_current['Date'].min() -  relativedelta(days=375)
        end_last_year = test_current['Date'].max() - relativedelta(days=350)

        tmp_train_time_ids = (pd.to_datetime(train['Date'])>start_last_year)&(pd.to_datetime(train['Date'])<end_last_year)
        tmp_train = train.loc[tmp_train_time_ids,]
        tmp_train = tmp_train.rename(columns={"Weekly_Sales": "Weekly_Pred"})

        test_pred = pd.merge(test_current,tmp_train,how='left',on=['Dept', 'Store', 'Wk'])
        # pred_ids = 


        # print(f'train:{train}')
    
    return train


def mypredict(train, test, next_fold, t):

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


    # print(f'test_current:{test_current}')
    holiday_ids = test_current['IsHoliday']==False
    test_current = test_current.loc[holiday_ids,]


    test_depts = test_current.Dept.unique()
    test_pred = None
    


    for dept in test_depts:    
    # no need to consider stores that do not need prediction
    # or do not have training samples
        train_dept_data = train[train['Dept']==dept]
        test_dept_data = test_current[test_current['Dept']==dept]
        train_stores = train_dept_data.Store.unique()
        test_stores = test_dept_data.Store.unique()
        test_stores = np.intersect1d(train_stores, test_stores)
    
        for store in test_stores:
            
            tmp_train = train_dept_data[train_dept_data['Store']==store]
            tmp_test = test_dept_data[test_dept_data['Store']==store]
            num_train = tmp_train.shape[0]
            num_test = tmp_test.shape[0]

            Wk_attribute_list = np.arange(1,53)
            # tmp_train['Wk'] = pd.Categorical(tmp_train['Wk'], categories=Wk_attribute_list)
            # tmp_test['Wk'] = pd.Categorical(tmp_test['Wk'], categories=Wk_attribute_list)

            train_weeks = train_dept_data.Wk.unique()
            # test_weeks = test_dept_data.Wk.unique()

            train_years = train_dept_data.Yr.unique()
            # test_years = test_dept_data.Yr.unique()

            trainWk_membership = [np.reshape(np.array(tmp_train['Wk'].values == elem).astype(np.float64), (num_train,1)) for elem in train_weeks]
            trainYr_intercept = [np.reshape(np.array(tmp_train['Yr'].values == elem).astype(np.float64), (num_train,1)) for elem in train_years]
            train_val_column =  np.reshape(tmp_train['Weekly_Sales'].values,(num_train,1))
            
            testWk_membership = [np.reshape(np.array(tmp_test['Wk'].values == elem).astype(np.float64), (num_test,1)) for elem in train_weeks]
            testYr_intercept = [np.reshape(np.array(tmp_test['Yr'].values == elem).astype(np.float64), (num_test,1)) for elem in train_years]

            trainDesign_matrix_a = np.hstack(tuple(trainWk_membership))
            trainDesign_matrix_b = np.hstack(tuple(trainYr_intercept))
            trainDesign_matrix = np.hstack(( trainDesign_matrix_a, trainDesign_matrix_b, train_val_column))

            testDesign_matrix_a = np.hstack(tuple(testWk_membership))
            testDesign_matrix_b = np.hstack(tuple(testYr_intercept))
            testDesign_matrix = np.hstack(( testDesign_matrix_a, testDesign_matrix_b))

            train_Y = trainDesign_matrix[:,-1]
            trainDesign_matrix = trainDesign_matrix[:,:-1]

            reg = LinearRegression().fit(trainDesign_matrix, train_Y)
            mycoef = reg.coef_
            myintercept = reg.intercept_

            mycoef[np.isnan(mycoef)] = 0
            if myintercept == np.nan:
                myintercept = 0

            tmp_pred = myintercept + np.dot(testDesign_matrix,mycoef).reshape(-1,1)
            tmp_test['Weekly_Pred'] = tmp_pred
            test_pred = pd.concat([test_pred, tmp_test])

    
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

    print('WAE:',wae)
    print(sum(wae)/len(wae))