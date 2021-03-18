import pandas as pd
import numpy as np
import datetime
from sklearn.impute import SimpleImputer
import dateutil.relativedelta
TOTAL_THRES = 300

def generate_label(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    cust = df[df['year_month']<year_month]['customer_id'].unique()
    df = df[df['year_month']==year_month]
    
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label

def feature_engineering2(df, year_month, show_score=True):
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=1)
    prev_ym = prev_ym.strftime('%Y-%m')
    
    train = df[df['year_month'] < prev_ym]
    val = df[df['year_month'] < year_month]
    
    train_label = generate_label(df, prev_ym)[['customer_id','year_month','label']]
    
    val_label = generate_label(df, year_month)[['customer_id','year_month','label']]
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        train_agg = train.loc[train['year_month'] < tr_ym].groupby(['customer_id']).agg(['mean', 'max', 'min', 'sum', 'count'])

        new_cols = []
        for col in train_agg.columns.levels[0]:
            for stat in train_agg.columns.levels[1]:
                new_cols.append(f'{col}-{stat}')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        if i == 0:
             all_train_data = train_agg.copy()
                
        else:
            all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    feature_names = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    val_agg = val.groupby(['customer_id']).agg(['mean', 'max', 'min', 'sum', 'count'])

    new_cols = []
    for col in val_agg.columns.levels[0]:
        for stat in val_agg.columns.levels[1]:
            new_cols.append(f'{col}-{stat}')

    val_agg.columns = new_cols
    
    val_data = val_label.merge(val_agg, on=['customer_id'], how='left')

    imputer = SimpleImputer(strategy='median')

    ft_tr = imputer.fit_transform(all_train_data.drop(columns=['customer_id', 'label', 'year_month']))
    ft_val = imputer.transform(val_data.drop(columns=['customer_id', 'label', 'year_month']))
    
    ft_tr = pd.DataFrame(ft_tr, columns=feature_names)
    ft_val = pd.DataFrame(ft_val, columns=feature_names)

    return ft_tr, ft_val, val_label, all_train_data, feature_names
