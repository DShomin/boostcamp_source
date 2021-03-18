import pandas as pd
import datetime
import dateutil.relativedelta
import os
from sklearn.impute import SimpleImputer
from utils import load_model, seed_everything
from features import feature_engineering2
from model import train_RF_model, train_lgb_model, get_importance

TOTAL_THRES = 300
SEED = 42
seed_everything(SEED)
DATA_PATH = os.environ.get('SM_CHANNEL_TRAIN', '/data')
MODEL_PATH = os.environ.get('SM_MODEL_DIR', '/model')
OUTPUT_PATH = os.environ.get('SM_OUTPUT_DATA_DIR', '/output')

if __name__ == '__main__':
    model_name = 'xgb'
    data = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    data['customer_id'] = data['customer_id'].astype(int)
    data['year_month'] = data.order_date.dt.strftime('%Y-%m')

    ## Create date inform feature
    data['weekday'] = data['order_date'].dt.weekday
    data['month'] = data['order_date'].dt.month
    data['minute'] = data['order_date'].dt.minute
    data['hour'] = data['order_date'].dt.hour
    data['date_only'] = data['order_date'].dt.date


    ft_tr, ft_val, val_label, all_train_data, feature_names = feature_engineering2(data, '2011-12')
    
    model = load_model(os.path.join(MODEL_PATH, f'{model_name}.pkl'))
    if model_name == 'RF':
        preds = model.predict_proba(ft_val)[:, 1]
    elif model_name == 'xgb':
        preds = model.predict(ft_val)
    elif model_name == 'ctb':
        preds = model.predict(ft_val)
    else:
        preds = model.predict(ft_val)

    sub = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
    sub['probability'] = preds
    sub.to_csv(os.path.join(OUTPUT_PATH, 'sub_test.csv'), index=False)