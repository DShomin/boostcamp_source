import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

import joblib

# Custom library
from utils import seed_everything, print_score, save_model
from features import generate_label, feature_engineering2
from model import train_RF_model, train_lgb_model, train_xgb_model, train_ctb_model, get_importance

TOTAL_THRES = 300
SEED = 42
seed_everything(SEED)
DATA_PATH = os.environ.get('SM_CHANNEL_TRAIN', '../input')
MODEL_PATH = os.environ.get('SM_MODEL_DIR', '../model')

if __name__ == '__main__':
    
    # train model_name
    model_name = 'xgb'

    # Load and base pre-process
    data = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), parse_dates=['order_date'])

    data['customer_id'] = data['customer_id'].astype(int)
    data['year_month'] = data.order_date.dt.strftime('%Y-%m')

    data = data[data['year_month']<'2011-12']
    data.reset_index(drop=True, inplace=True)

    ## Create date inform feature
    data['weekday'] = data['order_date'].dt.weekday
    data['month'] = data['order_date'].dt.month
    data['minute'] = data['order_date'].dt.minute
    data['hour'] = data['order_date'].dt.hour
    data['date_only'] = data['order_date'].dt.date
    ft_tr, ft_val, val_label, all_train_data, feature_names = feature_engineering2(data, '2011-11')
    
    if model_name == 'RF':
        model = train_RF_model(ft_tr, all_train_data['label'], model_name, MODEL_PATH)
        preds = model.predict_proba(ft_val)[:, 1]
    elif model_name == 'xgb':
        model = train_xgb_model(ft_tr, all_train_data['label'], model_name, MODEL_PATH)
        preds = model.predict(ft_val)
    elif model_name == 'ctb':
        model = train_ctb_model(ft_tr, all_train_data['label'], model_name, MODEL_PATH)
        preds = model.predict(ft_val)
    else:
        model = train_lgb_model(ft_tr, all_train_data['label'], model_name, MODEL_PATH)
        preds = model.predict(ft_val)
    
    fi = get_importance(model, feature_names, mode=model_name)


    print_score(val_label['label'], preds)