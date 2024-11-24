import sys
import os
import pickle
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
import optuna

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocessing import get_dataloader

current_dir = os.getcwd()  # Get the current directory
parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory

args = argparse.Namespace(
    sequence_len=30,
    feature_num=14,
    dataset_root=parent_dir + '/CMAPSSData/',
    max_rul=125,
    batch_size=128,
    lr=2e-3,
    step_size=10,
    gamma=0.1,
    weight_decay=1e-5,
    patience=8,
    max_epochs=30,
    use_exponential_smoothing=True,
    smooth_rate=40,
    no_cuda=False,
    save_model=False,
)

torch.manual_seed(28)

datasets = ['FD001', 'FD002', 'FD003', 'FD004']
metrics = {}

# Define hyperparameter grid for SVR model
svr_param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'epsilon': [0.1, 0.2, 0.5, 0.8],  # Epsilon for the margin of tolerance
    'kernel': ['linear', 'rbf'],  # Types of kernels
    'gamma': ['scale', 'auto']  # Kernel coefficient
}