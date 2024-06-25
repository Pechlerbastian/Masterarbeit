
import torch
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_percentage_error, r2_score
from scipy.stats import spearmanr
from collections import defaultdict

def _majority_target_Pitt(source_tag: list):
    return [re.match('.*-.*-', mark).group() for mark in source_tag]

def _majority_target_DAIC_WOZ(source_tag: list):
    return [mark.split('_')[0] for mark in source_tag]

def _majority_target_regression(source_tag: list):
    return ['regression_average'] * len(source_tag)

def majority_vote(source_tag: list, source_value: torch.Tensor, source_label: torch.Tensor, modify_tag, task='classification'):
    '''
    Args:
    source_tag: Guideline for voting, e.g. sample same.
    source_value: value before voting.
    source_label: label before voting.
    task: classification / regression

    Return:
    target: voting object.
    vote_value: value after voting.
    vote_label: label after voting.
    '''
    source_tag = modify_tag(source_tag)
    target = set(source_tag)
    vote_value_dict = {t:[] for t in target}
    vote_label_dict = {t:[] for t in target}

    if task == 'regression':
        logit_vote = True
    else:
        if source_value.dim() != 1:
            logit_vote = True
        else:
            logit_vote = False

    for i, (mark) in enumerate(source_tag):
        value = source_value[i]
        label = source_label[i]
        vote_value_dict[mark].append(value)
        vote_label_dict[mark].append(label)
    for key, value in vote_value_dict.items():
        if logit_vote:
            logit = torch.mean(torch.stack(value, dim=0), dim=0)
            if task == 'regression':
                vote_value_dict[key] = logit
            else:
                vote_value_dict[key] = torch.argmax(logit)
        else:
            vote_value_dict[key] = max(value, key=value.count)

    vote_value, vote_label = [], []
    for t in target:
        vote_value.append(vote_value_dict[t])
        vote_label.append(vote_label_dict[t][0])

    vote_value = torch.tensor(vote_value)
    vote_label = torch.tensor(vote_label)
    
    return target, vote_value, vote_label

def calculate_score_classification(preds, labels, average_f1='weighted'):  # weighted, macro
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_f1, zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, ua, f1, precision, confuse_matrix

def calculate_score_regression(preds, labels, ids=None):
    mse = mean_squared_error(y_true=labels, y_pred=preds, squared=True)
    rmse = mean_squared_error(y_pred=preds, y_true=labels, squared=False)
    mae = (preds - labels).abs().mean()
    average_spearman = None

    if ids != None:
        spearman_per_person, _ = calculate_spearman_per_id(preds, labels, ids)
    
        average_spearman = np.mean(np.asarray(spearman_per_person))

    # Calculate overall Spearman coefficient
    all_labels = np.array(labels)
    all_predictions = np.array(preds)
    overall_spearman, _ = spearmanr(all_labels, all_predictions) 

    return mse, mae, rmse, average_spearman, overall_spearman

def calculate_spearman_per_id(preds, labels, ids):
    group_data = {}
    
    # Organize data per id
    for prediction, label, subject_id in zip(preds, labels, ids):
        if subject_id in group_data:
            group_data[subject_id].append((label.item(), prediction.item()))
        else:
            group_data[subject_id] = [(label.item(), prediction.item())]

    # Calculate Spearman coefficient per group
    spearman_coefficients = []
    corresponding_ids = []

    for subject_id, data in group_data.items():
        labels_group, predictions_group = zip(*data)
        spearman_coefficient, _ = spearmanr(labels_group, predictions_group)
        if not np.isnan(spearman_coefficient):
            spearman_coefficients.append(spearman_coefficient)
            corresponding_ids.append(subject_id)
        else:
            print(subject_id)

    print(spearman_coefficients)
    # Calculate average Spearman coefficient
    return spearman_coefficients, corresponding_ids

def calculate_basic_score_regression(preds, labels):
    #for regression
    return mean_squared_error(y_pred=preds, y_true=labels)
    # spearman_per_person, _ = calculate_spearman_per_id(preds, labels, ids)
    # return np.mean(spearman_per_person)

def calculate_basic_score(preds, labels):
    #for classification
    return accuracy_score(labels, preds)

def tidy_csvfile(csvfile, colname, ascending=True):
    '''
    tidy csv file base on a particular column.
    '''
    print(f'tidy file: {csvfile}, base on column: {colname}')
    df = pd.read_csv(csvfile)
    df = df.sort_values(by=[colname], ascending=ascending, na_position='last')
    df = df.round(3)
    df.to_csv(csvfile, index=False, sep=',')

    
