import numpy as np
import pandas as pd
from datetime import date
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

def mypredict(train, test, next_fold, t):
    
    
    if t!=1:
        train = pd.concat([train,next_fold])
    start_date = pd.to_datetime("2011-03-01") + relativedelta(months=2 * (t-1))
    end_date = pd.to_datetime("2011-05-01") + relativedelta(months=2 * (t-1))

    find_week = lambda x : x.isocalendar()[1]

    test['Wk'] =  pd.to_datetime(test['Date']).apply(find_week)
    train['Wk'] =  pd.to_datetime(train['Date']).apply(find_week)

    time_ids = (pd.to_datetime(test['Date'])>=start_date)&(pd.to_datetime(test['Date'])<end_date)
    test_current = test.loc[time_ids,]

    start_last_year = pd.to_datetime(test_current['Date'].min()) -  relativedelta(days=375)
    end_last_year = pd.to_datetime(test_current['Date'].max()) - relativedelta(days=350)


    tmp_train_time_ids = (pd.to_datetime(train['Date'])>start_last_year)&(pd.to_datetime(train['Date'])<end_last_year)
    tmp_train = train.loc[tmp_train_time_ids,]

    tmp_train = tmp_train.rename(columns={'Weekly_Sales':'Weekly_Pred'})
    tmp_train = tmp_train.drop(['Date'],axis = 1)

    test_pred = test_current.merge(tmp_train,how='left',on=['Dept', 'Store', 'Wk'])
    test_pred.drop(['Wk'],axis=1)

    train = train.drop(['Wk'],axis=1)

    

    
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
        # print(next_fold.head(5))

        # extract predictions matching up to the current fold
        scoring_df = next_fold.merge(test_pred, on=['Date', 'Store', 'Dept'], how='left')

        # extract weights and convert to numpy arrays for wae calculation
        weights = scoring_df['IsHoliday_x'].apply(lambda is_holiday:5 if is_holiday else 1).to_numpy()
        actuals = scoring_df['Weekly_Sales'].to_numpy()
        preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

        wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

    print('WAE:',wae)
    print(sum(wae)/len(wae))