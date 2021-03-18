from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from utils import save_model
import pandas as pd
import os

# SEED
def train_RF_model(ft_tr, label, model_name, save_path):
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(ft_tr, label) # all_train_data['label']
    save_model(model, save_path + f'{model_name}.pkl')
    return model

def train_lgb_model(ft_tr, label, model_name, save_path):
    dtrain = lgb.Dataset(ft_tr, label=label)
        
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'n_estimators': 100,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,    
    } 

    model = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dtrain],
        verbose_eval=50,
    )
    
    save_model(model, os.path.join(save_path, f'{model_name}.pkl'))

    return model

def train_xgb_model(ft_tr, label, model_name, save_path):
    
    xgb_params = {
        'objective': 'binary:logistic',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'n_estimators': 100,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,  
    }
    
    model = xgb.XGBClassifier() # **xgb_params
    model.fit(ft_tr, label)
    save_model(model, os.path.join(save_path, f'{model_name}.pkl'))

    return model

def train_ctb_model(ft_tr, label, model_name, save_path):
    cat_params = {
        'n_estimators':1000,
        'learning_rate': 0.07,
        'eval_metric':'AUC',
        'loss_function':'Logloss',
        'random_seed':42,
        'metric_period':500,
        'od_wait':500,
        'depth': 8,
        #'colsample_bylevel':0.7,
    } 

    model = ctb.CatBoostClassifier(**cat_params) # **ctb_params
    model.fit(ft_tr, label)
    save_model(model, os.path.join(save_path, f'{model_name}.pkl'))
    return model

def get_importance(model, feature_names, mode='RF'):
    if mode == 'RF' or mode == 'xgb' or mode == 'ctb':
        fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    else:
        fi = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importance()})

    return fi