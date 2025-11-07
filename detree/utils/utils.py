import hashlib
import os
import pickle
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score,roc_curve

def stable_long_hash(input_string):
    hash_object = hashlib.sha256(input_string.encode())
    hex_digest = hash_object.hexdigest()
    int_hash = int(hex_digest, 16)
    long_long_hash = (int_hash & ((1 << 63) - 1))
    return long_long_hash

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)



def find_top_n(embeddings,n,index,data):
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    top_ids_and_scores = index.search_knn(embeddings, n)
    data_ans=[]
    for i, (ids, scores) in enumerate(top_ids_and_scores):
        data_now=[]
        for id in ids:
            data_now.append((data[0][int(id)],data[1][int(id)],data[2][int(id)]))
        data_ans.append(data_now)
    return data_ans


    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_line(class_name, metrics, is_header=False):
    if is_header:
        line = f"| {'Class':<10} | " + " | ".join([f"{metric:<10}" for metric in metrics])
    else:
        line = f"| {class_name:<10} | " + " | ".join([f"{metrics[metric]:<10.3f}" for metric in metrics])
    print(line)
    if is_header:
        print('-' * len(line))

def calculate_per_class_metrics(classes, ground_truth, predictions):
    # Convert ground truth and predictions to numeric format
    gt_numeric = np.array([int(gt) for gt in ground_truth])
    pred_numeric = np.array([int(pred) for pred in predictions])

    results = {}
    for i, class_name in enumerate(classes):
        # For each class, calculate the 'vs rest' binary labels
        gt_binary = (gt_numeric == i).astype(int)
        pred_binary = (pred_numeric == i).astype(int)

        # Calculate metrics, handling cases where a class is not present in predictions or ground truth
        precision = precision_score(gt_binary, pred_binary, zero_division=0)
        recall = recall_score(gt_binary, pred_binary, zero_division=0)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)
        acc = np.mean(gt_binary == pred_binary)
        # Calculate recall for all other classes as 'rest'
        rest_recall = recall_score(1 - gt_binary, 1 - pred_binary, zero_division=0)

        results[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': acc,
            'Avg Recall (with rest)': (recall + rest_recall) / 2
        }

    print_line("Metric", results[classes[0]], is_header=True)
    for class_name, metrics in results.items():
        print_line(class_name, metrics)
    overall_metrics = {metric_name: np.mean([metrics[metric_name] for metrics in results.values()]) for metric_name in results[classes[0]].keys()}
    print_line("Overall", overall_metrics)

def calculate_metrics(labels, preds):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return acc, precision, recall, f1

def compute_three_recalls(labels, preds):
    all_n, all_p, tn, tp = 0, 0, 0, 0
    for label, pred in zip(labels, preds):
        if label == '0':
            all_p += 1
        if label == '1':
            all_n += 1
        # Modified condition to treat None in preds as incorrect prediction
        if pred is not None and label == pred == '0':
            tp += 1 
        # Modified condition to treat None in preds as incorrect prediction
        if pred is not None and label == pred == '1':
            tn += 1
        if pred is None:
            continue
    machine_rec , human_rec= tp * 100 / all_p if all_p != 0 else 0, tn * 100 / all_n if all_n != 0 else 0
    avg_rec = (human_rec + machine_rec) / 2
    return (human_rec, machine_rec, avg_rec)


def compute_metrics(labels, preds,ids=None):
    # Handling None values in preds as incorrect predictions
    #preds = ['0' if pred is None else pred for pred in preds]
    if ids is not None:
        # Deduplicate labels and predictions for repeated ids
        dict_labels,dict_preds={},{}
        for i in range(len(ids)):
            dict_labels[ids[i]]=labels[i]
            dict_preds[ids[i]]=preds[i] 
        labels=list(dict_labels.values())
        preds=list(dict_preds.values())
    
    human_rec, machine_rec, avg_rec = compute_three_recalls(labels, preds)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, pos_label='1')
    recall = recall_score(labels, preds, pos_label='1')
    f1 = f1_score(labels, preds, pos_label='1')
    # return human_rec, machine_rec, avg_rec
    return (human_rec, machine_rec, avg_rec, acc, precision, recall, f1)

def evaluate_max_f1_metrics(test_labels, y_score):
    test_labels = np.array(test_labels)
    y_score = np.array(y_score)

    auroc = roc_auc_score(test_labels, y_score)
    precision, recall, thresholds = precision_recall_curve(test_labels, y_score, pos_label=1)
    pr_auc = auc(recall, precision)
    epsilon = 1e-6
    f1_scores = 2 * precision * recall / (precision + recall+epsilon)
    best_index = f1_scores.argmax()
    best_f1 = f1_scores[best_index]
    best_precision = precision[best_index]
    best_recall = recall[best_index]
    
    threshold = thresholds[best_index] if best_index < len(thresholds) else 1.0
    y_pred_max_f1 = (y_score >= threshold).astype(int)
    
    acc = (y_pred_max_f1 == test_labels).mean()
    tp = sum((y_pred_max_f1 == 1) & (test_labels == 1))
    fn = sum((y_pred_max_f1 == 0) & (test_labels == 1))
    fp = sum((y_pred_max_f1 == 1) & (test_labels == 0))
    tn = sum((y_pred_max_f1 == 0) & (test_labels == 0))

    pos_recall = tp / (tp + fn + epsilon)  # recall for the positive class
    neg_recall = tn / (tn + fp + epsilon)  # recall for the negative class
    avg_recall = (pos_recall + neg_recall) / 2  # average recall across classes
    
    metric = {'auroc': auroc, 'pr_auc': pr_auc, 'F1': best_f1, 'Precision': best_precision,\
               'Recall': best_recall, 'threshold': threshold, 'acc': acc, 'avg_recall': avg_recall,\
                  'pos_recall': pos_recall, 'neg_recall': neg_recall}
    return metric

def evaluate_metrics(test_labels, y_score, threshold_param=-1,target_fpr = 0.05):
    if isinstance(test_labels, list):
        test_labels = np.array(test_labels)
    if isinstance(y_score, list):
        y_score = np.array(y_score)

    if threshold_param != -1:
        if not (0 <= threshold_param <= 1):
            raise ValueError("Threshold must be between 0 and 1.")

    auroc = roc_auc_score(test_labels, y_score)

    precision, recall, thresholds = precision_recall_curve(test_labels, y_score, pos_label=1)
    pr_auc = auc(recall, precision)

    epsilon = 1e-6
    f1_scores = 2 * precision * recall / (precision + recall + epsilon)
    

    if threshold_param == -1:
        best_index = f1_scores.argmax()
        F1 = f1_scores[best_index]
        Precision = precision[best_index]
        Recall = recall[best_index]
        threshold = thresholds[best_index] if best_index < len(thresholds) else 1.0
    else:
        threshold = threshold_param
        index = np.where(thresholds >= threshold)[0][0]
        Precision = precision[index]
        Recall = recall[index]
        F1 = f1_scores[index]


    y_pred = (y_score >= threshold).astype(int)
    acc = (y_pred == test_labels).mean()
    
    tp = ((y_pred == 1) & (test_labels == 1)).sum()
    fn = ((y_pred == 0) & (test_labels == 1)).sum()
    fp = ((y_pred == 1) & (test_labels == 0)).sum()
    tn = ((y_pred == 0) & (test_labels == 0)).sum()

    pos_recall = tp / (tp + fn + epsilon)  # TPR
    neg_recall = tn / (tn + fp + epsilon)  # TNR
    avg_recall = (pos_recall + neg_recall) / 2

    fpr, tpr, thds = roc_curve(test_labels, y_score)
    if len(fpr) > 0 and len(tpr) > 0:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr = tpr[idx]
        tpr_at_fpr_threshold = thds[idx]
    else:
        tpr_at_fpr = 0.0

    metric = {'auroc': auroc, 'pr_auc': pr_auc, 'F1': F1, 'Precision': Precision,'Recall': Recall,\
               'threshold': threshold, 'acc': acc, 'avg_recall': avg_recall,'pos_recall': pos_recall,\
                  'neg_recall': neg_recall, 'tpr_at_fpr': tpr_at_fpr, 'tpr_at_fpr_threshold': tpr_at_fpr_threshold}
    
    return metric
    # return (auroc, pr_auc, best_f1, best_precision, best_recall, threshold,
    #         acc, avg_recall, pos_recall, neg_recall, tpr_at_fpr5)


def load_datapath(path,include_adversarial=False,dataset_name='all',attack_type='all'):
    data_path = {'train':[],'valid':[],'test':[]}
    if dataset_name=='all':
        datasets = os.listdir(path)
    elif dataset_name=='M4':
        datasets = ['M4_monolingual','M4_multilingual']
    elif dataset_name=='RAID_all':
        datasets = ['RAID','RAID_extra']
    else:
        datasets = [dataset_name]
    for dataset in datasets:
        dataset_path = os.path.join(path,dataset)
        if attack_type!='all':
            dataset_path_list = [pth for pth in os.listdir(dataset_path) if attack_type in pth]
        else:
            dataset_path_list = os.listdir(dataset_path)
        for adv in dataset_path_list:
            if include_adversarial==False and 'no_attack' not in adv:
                continue
            adv_path = os.path.join(dataset_path,adv)
            for data in os.listdir(adv_path):
                if 'train.' in data:
                    data_path['train'].append(os.path.join(adv_path,data))
                elif 'test.' in data:
                    data_path['test'].append(os.path.join(adv_path,data))
                elif 'valid.' in data:
                    data_path['valid'].append(os.path.join(adv_path,data))
    return data_path