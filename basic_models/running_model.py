import sys
import os
import pickle
import argparse
import  torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import matplotlib.pyplot as plt

import numpy as np


# import linear regression, svm and random forest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# import rmse and mse and r2 score
from sklearn.metrics import mean_squared_error, r2_score
# imprort rmse
from sklearn.metrics import mean_squared_error

from preprocessing import get_dataloader

if __name__ == '__main__':
    current_dir = os.getcwd()  # Get the current directory
    parent_dir = os.path.dirname(current_dir)  # Get the upper-level directory
    parser = argparse.ArgumentParser(description='Cmapss Dataset With Pytorch')

    parser.add_argument('--sequence-len', type=int, default=30)
    parser.add_argument('--feature-num', type=int, default=14)
    parser.add_argument('--dataset-root', type=str,
                        default=parent_dir + '/CMAPSSData/',
                        help='The dir of CMAPSS dataset1')
    parser.add_argument('--max-rul', type=int, default=125, help='piece-wise RUL')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--step-size', type=int, default=10, help='interval of learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='ratio of learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8, help='Early Stop Patience')
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--use-exponential-smoothing', default=True)
    parser.add_argument('--smooth-rate', type=int, default=40)
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save-model', type=str, default=False, help='save trained models')
    args = parser.parse_args()

    torch.manual_seed(28)

    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    metrics = {}

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        args.sub_dataset = dataset  # Set current dataset

        train_loader, valid_loader, test_loader, test_loader_last, \
            num_test_windows, train_visualize, engine_id = get_dataloader(
                dir_path=args.dataset_root,
                sub_dataset=args.sub_dataset,
                max_rul=args.max_rul,
                seq_length=args.sequence_len,
                batch_size=args.batch_size,
                use_exponential_smoothing=args.use_exponential_smoothing,
                smooth_rate=args.smooth_rate)

        # Fit linear regression model
        linear = LinearRegression()
        for i, (x, y) in enumerate(train_loader):
            x = x.view(-1, args.sequence_len * args.feature_num).numpy()
            y = y.numpy()
            linear.fit(x, y)

        # Validate and Test
        valid_predictions, valid_actuals = [], []
        test_predictions, test_actuals = [], []

        # Validation set
        for i, (x, y) in enumerate(valid_loader):
            x = x.view(-1, args.sequence_len * args.feature_num).numpy()
            y = y.numpy()
            preds = linear.predict(x)

            valid_predictions.extend(preds)
            valid_actuals.extend(y)

        # Test set
        for i, (x, y) in enumerate(test_loader):
            x = x.view(-1, args.sequence_len * args.feature_num).numpy()
            y = y.numpy()
            preds = linear.predict(x)

            test_predictions.extend(preds)
            test_actuals.extend(y)

        # Post-process predictions
        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)
        valid_predictions = np.array(valid_predictions)
        valid_actuals = np.array(valid_actuals)

        max_rul = max(test_actuals.max(), valid_actuals.max())
        test_predictions = np.clip(test_predictions, 0, max_rul)
        valid_predictions = np.clip(valid_predictions, 0, max_rul)
        test_predictions = test_predictions * args.max_rul
        test_actuals = test_actuals * args.max_rul

        # Compute Metrics
        valid_mse = mean_squared_error(valid_actuals, valid_predictions)
        valid_rmse = np.sqrt(valid_mse)
        valid_r2 = r2_score(valid_actuals, valid_predictions)

        test_mse = mean_squared_error(test_actuals, test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(test_actuals, test_predictions)

        metrics[dataset] = {
            "Validation MSE": valid_mse,
            "Validation RMSE": valid_rmse,
            "Validation R2": valid_r2,
            "Test MSE": test_mse,
            "Test RMSE": test_rmse,
            "Test R2": test_r2
        }

        print(f"Dataset {dataset} Metrics:")
        print(f"Validation MSE: {valid_mse:.4f}, RMSE: {valid_rmse:.4f}, R2: {valid_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        # Plot predictions vs actuals
        plt.figure(figsize=(10, 6))
        plt.plot(test_actuals, label='Actual RUL')
        plt.plot(test_predictions, label='Predicted RUL')
        plt.xlabel('Window')
        plt.ylabel('RUL')
        plt.title(f'Linear Regression Model - {dataset}')
        plt.legend()
        plt.savefig(f"{dataset}_rul_plot.png")  # Save plot
        plt.show()

    # Print summary of metrics
    print("\nSummary of Metrics:")
    for dataset, dataset_metrics in metrics.items():
        print(f"{dataset}:")
        for metric_name, value in dataset_metrics.items():
            print(f"  {metric_name}: {value:.4f}")