import pandas as pd
from sklearn.metrics import roc_auc_score
import os

def evaluation(gt_path, pred_path):
    '''
    Args:
        gt_path (string) : ropot directory of ground truth file
        pred_path (string) : root directory of prediction file (output of inference.py)
    '''

    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)

    score = roc_auc_score(gt['label'], pred['probability'])

    return f'{score:.4f}%'

if __name__ == '__main__':
    gt_path = os.environ.get('SM_GROUND_TRUTH_DIR')
    pred_path = os.environ.get('SM_OUTPUT_DATA_DIR')

    result_str = evaluation(gt_path, pred_path)
    print(f'Final Score: {result_str}')