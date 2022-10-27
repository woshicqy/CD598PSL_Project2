import pandas as pd
from dateutil.relativedelta import relativedelta

# read raw data and extract date column
train_raw = pd.read_csv('train.csv')

# training data from 2010-02 to 2011-02
start_date = pd.to_datetime('2010-02-01')
end_date = start_date + relativedelta(months=13)

# split dataset into training / testing
train_ids = (pd.to_datetime(train_raw['Date']) >= start_date) & (pd.to_datetime(train_raw['Date']) < end_date)
train = train_raw.loc[train_ids, ]
test = train_raw.loc[~train_ids, ]

# create the initial training data
train.to_csv('train_ini.csv')

# create test.csv 
# removes weekly sales
test = test.drop(columns=['Weekly_Sales'])
test.to_csv('test.csv')

# create 10 time-series
num_folds = 10

# month 1 --> 2011-03, and month 20 --> 2012-10.
# Fold 1 : month 1 & month 2, Fold 2 : month 3 & month 4 ...
for i in range(num_folds):
  # filter fold for dates
  start_date = pd.to_datetime('2011-03-01') + relativedelta(months = 2 * i)
  end_date = pd.to_datetime('2011-05-01') + relativedelta(months = 2 * i)
  test_ids = (pd.to_datetime(test['Date']) >= start_date) & (pd.to_datetime(test['Date']) < end_date)
  test_fold = test.loc[test_ids, ]

  # write fold to a file
  test_fold.to_csv('fold_{}.csv'.format(i + 1))
  print(f'Folds:{i+1} is created....')

print('Created is Done!')