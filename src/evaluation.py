import pandas as pd
from sklearn.metrics import roc_auc_score

# 평가 함수
def evaluation(gt_path, pred_path):
    # Ground Truth 경로에서 정답 파일 읽기
    label = pd.read_csv(gt_path + '/label.csv')['label']
    
    # 테스트 결과 예측 파일 읽기
    preds = pd.read_csv(pred_path + '/submission.csv')['probability']

    # AUC 스코어 계산
    score = roc_auc_score(label, preds)
    
    return score


if __name__ == '__main__':
    gt_path = '../input'
    pred_path = '.'
    
    print(evaluation(gt_path, pred_path))
    