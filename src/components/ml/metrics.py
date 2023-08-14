import pandas as pd 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc

def evaluate_model_kpi(model, X_train, y_train, X_val, y_val, threshold=0.5, model_name=None):
    ''' 
    Area under the curve-ROC (AUC-ROC): 
    The measure of the area under the Receiver Operating Characteristic curve, indicating the classifier's ability to distinguish between classes across various threshold values.
    
    Area under the curve-PRC (AUC-PRC): 
    The measure of the area under the Precision-Recall curve, illustrating the trade-off between precision and recall for different classification thresholds.
    
    Accuracy: 
    The proportion of correctly predicted instances to the total number of instances.
    (TP + TN) / (TP + TN + FP + FN).
    
    Precision: 
    The ratio of true positive predictions to the total number of positive predictions, quantifying the model's accuracy in predicting positive cases.
    TP / (TP + FP).
    
    Recall: 
    The ratio of true positive predictions to the total number of actual positives, reflecting the model's ability to capture all positive instances.
    TP / (TP + FN).
    
    F1 Score: 
    The harmonic mean of precision and recall, combining both metrics to provide a balanced evaluation of a model's performance on positive class prediction.
    2 * (Precision * Recall) / (Precision + Recall).
    '''   

    if model_name == None:
        model_name = type(model)
    
    # Make predictions using the model and the DataFrame
    y_train_pred = (model.predict_proba(X_train)[:,1] >= threshold).astype(bool)
    y_val_pred = (model.predict_proba(X_val)[:,1] >= threshold).astype(bool)
    
    y_train_prob = model.predict_proba(X_train)[:,1]
    y_val_prob = model.predict_proba(X_val)[:,1]
    
    # Calculate accuracy
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    
    # Calculate precision, recall, and f1 score
    prf_train = precision_recall_fscore_support(y_train, y_train_pred, average=None, zero_division=1)
    prf_val = precision_recall_fscore_support(y_val, y_val_pred, average=None, zero_division=1)
    
    # Calculate area under the curve
    auc_train = roc_auc_score(y_train, y_train_prob)
    auc_val = roc_auc_score(y_val, y_val_prob)
    
    # Calculate area under the precision-recall curve
    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, y_train_pred)
    auc_precision_recall_train = auc(recall_train, precision_train)
    
    precision_val, recall_val, thresholds_train = precision_recall_curve(y_val, y_val_prob)
    auc_precision_recall_val = auc(recall_val, precision_val)
    
    # Store the metrics in a DataFrame
    metrics = pd.DataFrame({
        'Algorithm': model_name,
         
        'AUC-ROC Train': [auc_train],
        'AUC-ROC Val': [auc_val],
        'AUC-PRC Train': [auc_precision_recall_train],
        'AUC-PRC Val': [auc_precision_recall_val],
        
        'Accuracy Train': [accuracy_train],
        'Accuracy Val': [accuracy_val],
        
        'Precision Train: 0': [prf_train[0][0]],
        'Precision Val: 0': [prf_val[0][0]],
        'Precision Train: 1': [prf_train[0][1]],
        'Precision Val: 1': [prf_val[0][1]],
        
        'Recall Train: 0': [prf_train[1][0]],
        'Recall Val: 0': [prf_val[1][0]],
        'Recall Train: 1': [prf_train[1][1]],
        'Recall Val: 1': [prf_val[1][1]],
        
        'F1-score Train: 0': [prf_train[2][0]],
        'F1-score Val: 0': [prf_val[2][0]],
        'F1-score Train: 1': [prf_train[2][1]],
        'F1-score Val: 1': [prf_val[2][1]],
    })
    
    return metrics