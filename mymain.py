# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import date
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, ensemble
import warnings
warnings.filterwarnings("ignore")

def convert(x,drop=True):
    find_week = lambda x : x.isocalendar()[1]
    find_yr = lambda x : x.isocalendar()[0]
    quadratic_Yr = lambda x : (x.isocalendar()[0])**2

    # x_edited = x.drop(['Date'], axis=1)
    x_edited = x.copy()
    x_edited['Wk'] = x['Date'].apply(find_week)
    x_edited['Yr'] = x['Date'].apply(find_yr)
    if drop:
      return x_edited[['Store','Dept', 'Yr', 'Wk', 'IsHoliday','Date']]
    else:
      return x_edited[['Store','Dept', 'Yr', 'Wk', 'Weekly_Sales','Date']]

def mypredict(train, test, next_fold, t_input):

    t_test = t_input
    # mypredict happens

    # t for python starts from 0
    t = t_input - 1
    
    if type(test) != None:
        train = pd.concat([train,next_fold],ignore_index=True)

    # not all depts need prediction
    
    start_date = pd.to_datetime("2011-03-01") + relativedelta(months=2 * (t))
    end_date = pd.to_datetime("2011-05-01") + relativedelta(months=2 * (t))
    
    time_ids = (pd.to_datetime(test['Date'])>=start_date)&(pd.to_datetime(test['Date'])<end_date)
    test = test.loc[time_ids,].reset_index(drop=True)

    if t != t_test-1:
          return train, test
    
    if t == t_test-1:
      test = convert(test)
      converted = convert(train,False)

      test_current = test.copy()
      test_depts = test_current.Dept.unique()
      test_pred = None

      holiday = [36, 47, 52, 6]
      
      converted['time'] = converted['Yr'] * 100 + converted['Wk']

      x_gb = converted.groupby(['Dept'])
      inds = np.array([(x_gb.get_group(i))[['Dept']].iloc[0].to_list() for i in x_gb.groups])
      converted_splitted = [(x_gb.get_group(i)).drop(['Dept'], axis=1) for i in x_gb.groups]

      inds = np.array(inds).reshape(-1)

      cropped_df = []
      d = 10
      for i in range(inds.size):

        X = (converted_splitted[i]).drop(['Yr'], axis=1).pivot(
            index='Store', columns='time', values='Weekly_Sales').fillna(0)
        X_mean = X.mean(axis=1).to_numpy()

        X_np = X.to_numpy()
        u, s, vh = np.linalg.svd((X_np.T - X_mean).T, full_matrices=False)
        s_cropped = np.diag(np.array([s[i] if i < d else 0 for i in range(s.size)]))

        X_pca = ((u @ s_cropped @ vh).T + X_mean).T

        #put cropped X back to df
        X_cropped = pd.DataFrame(X_pca, index=X.index, columns=X.columns)

        X_cropped = X_cropped.reset_index()
        X1 = pd.melt(X_cropped, id_vars='Store', value_vars=X_cropped.columns).sort_values(by=['Store','time'])
        final_df = pd.DataFrame([])

        final_df['Store'] = X1['Store']
        final_df['Yr'] = np.floor(X1['time'].to_numpy().astype(int)/100).astype(int)
        final_df['Wk'] = np.mod(X1['time'].to_numpy().astype(int), 100)
        final_df['Weekly_Sales'] = pd.melt(X_cropped, id_vars='Store', value_vars=X_cropped.columns)['value']

        final_df = final_df.reindex()
        cropped_df.append(final_df)

      for i in range(inds.size):
          set_size = cropped_df[i].shape[0]
          cropped_df[i]['Dept'] = (np.ones(set_size)*inds[i]).astype(int)
      
      output = pd.concat(cropped_df).reset_index()

      holiday = [36, 47, 52, 6]
      wk = output['Wk']
      output['IsHoliday'] = np.where((wk.isin(holiday)).to_numpy(),True,False)

      for dept in test_depts:
        train_dept_data = output[output['Dept']==dept]
        test_dept_data = test_current[test_current['Dept']==dept]
        train_stores = train_dept_data.Store.unique()
        test_stores = test_dept_data.Store.unique()
        test_stores = np.intersect1d(train_stores, test_stores)

        for store in test_stores:
          tmp_train = train_dept_data[train_dept_data['Store']==store]
          tmp_test = test_dept_data[test_dept_data['Store']==store]

          # print(tmp_train.head(5))
          trainY = tmp_train['Weekly_Sales']
          tmp_train = tmp_train.drop(['Weekly_Sales'],axis=1)

          ohe = OneHotEncoder(handle_unknown='ignore',sparse=False,drop='if_binary')
          enc = ohe.fit_transform(tmp_train[['Wk','IsHoliday','Yr']])

          ### encoding features ###
          train_dummy = pd.DataFrame(enc,columns=ohe.get_feature_names_out())
          test_dummy = pd.DataFrame(ohe.transform(tmp_test[['Wk','IsHoliday','Yr']]),columns=ohe.get_feature_names_out())

          ### covert to dataframe ###
          train_dummy['Store'] = tmp_train['Store'].to_numpy()
          train_dummy['Dept'] = tmp_train['Dept'].to_numpy()

          test_dummy['Store'] = tmp_test['Store'].to_numpy()
          test_dummy['Dept'] = tmp_test['Dept'].to_numpy()

          new_col_list = ohe.get_feature_names_out()
          new_col_list = new_col_list.tolist() + ['Dept','Store']
          train_dummy = train_dummy[new_col_list]
          test_dummy = test_dummy[new_col_list]
          
          reg = Ridge(alpha = 0.17)
          reg.fit(train_dummy, trainY)

          tmp_pred = reg.predict(test_dummy)
          tmp_test['Weekly_Pred'] = tmp_pred
          #tmp_test = tmp_test.drop(['Wk','Yr'],axis=1)

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
      ae = (np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item()
      print('AE:',ae)

      wae.append((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())
      # print((np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())
  print('WAE:',wae)
  print(sum(wae)/len(wae))

