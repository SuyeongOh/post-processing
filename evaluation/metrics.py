import csv
import os
import itertools
import numpy as np
import torch
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
from evaluation.post_process import calculate_metric_per_video


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def _reform_data_from_dict(data, flatten=True):
    """Reformat predictions and labels from dicts."""
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())
    return sort_data


def initialize_csv(csv_path):
    """Initialize CSV file with headers if it doesn't exist."""
    headers = ['EVALUATION_METHOD', 'MODEL', 'POST_PROCESSING', 'MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def get_test_conditions(config):
    """Generate all combinations of test conditions using Cartesian product."""
    methods = config.INFERENCE.EVALUATION_METHOD
    bandpass_opts = [True, False]
    diff_opts = [True, False]
    detrend_opts = [True, False]

    # (Method, BP, Diff, Detrend) 튜플의 리스트 생성
    return itertools.product(methods, bandpass_opts, diff_opts, detrend_opts)


# ---------------------------------------------------------
# Core Logic Functions
# ---------------------------------------------------------

def process_dataset_under_condition(predictions, labels, config, condition):
    """
    Process the entire dataset under a specific test condition.
    condition: tuple (eval_method, use_bandpass, diff_flag, detrend_flag)
    """
    eval_method, use_bandpass, diff_flag, detrend_flag = condition

    # Method Name Normalization for internal function
    if eval_method == "peak detection":
        hr_method_arg = "Peak"
    elif eval_method == "FFT":
        hr_method_arg = "FFT"
    else:
        hr_method_arg = eval_method

    gt_hr_all, predict_hr_all, SNR_all = [], [], []

    desc_str = f"{eval_method[:4]} (BP={int(use_bandpass)}, Diff={int(diff_flag)}, Det={int(detrend_flag)})"

    for index in tqdm(predictions.keys(), ncols=80, desc=desc_str):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        # Window Size Logic
        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            window_frame_size = min(window_frame_size, video_frame_size)
        else:
            window_frame_size = video_frame_size

        # Window Loop
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i + window_frame_size]
            label_window = label[i:i + window_frame_size]

            if len(pred_window) < 9:
                continue

            gt_hr, pred_hr, snr, _ = calculate_metric_per_video(
                pred_window,
                label_window,
                diff_flag=diff_flag,
                fs=config.TEST.DATA.FS,
                hr_method=hr_method_arg,
                use_bandpassfilter=use_bandpass,
                detrend_flag=detrend_flag
            )

            gt_hr_all.append(gt_hr)
            predict_hr_all.append(pred_hr)
            SNR_all.append(snr)

    return np.array(gt_hr_all), np.array(predict_hr_all), np.array(SNR_all)


def compute_statistics(gt, pred, snr, metrics_list):
    """Compute statistical metrics based on results."""
    stats = {}
    num_samples = len(pred)

    for metric in metrics_list:
        if metric == "MAE":
            val = np.mean(np.abs(pred - gt))
            std = np.std(np.abs(pred - gt)) / np.sqrt(num_samples)
            stats["MAE"] = (val, std)
        elif metric == "RMSE":
            val = np.sqrt(np.mean(np.square(pred - gt)))
            std = np.std(np.square(pred - gt)) / np.sqrt(num_samples)
            stats["RMSE"] = (val, std)
        elif metric == "MAPE":
            val = np.mean(np.abs((pred - gt) / gt)) * 100
            std = np.std(np.abs((pred - gt) / gt)) / np.sqrt(num_samples) * 100
            stats["MAPE"] = (val, std)
        elif metric == "Pearson":
            corr = np.corrcoef(pred, gt)
            val = corr[0][1]
            std = np.sqrt((1 - val ** 2) / (num_samples - 2))
            stats["Pearson"] = (val, std)
        elif metric == "SNR":
            val = np.mean(snr)
            std = np.std(snr) / np.sqrt(num_samples)
            stats["SNR"] = (val, std)

    return stats


def save_results_to_csv(csv_path, config, condition, stats):
    """Format and append results to CSV."""
    eval_method, use_bandpass, diff_flag, detrend_flag = condition

    # Post Processing String
    pp_list = [f"DiffNormalized={diff_flag}", f"Bandpass={use_bandpass}", f"Detrend={detrend_flag}"]
    pp_str = " | ".join(pp_list)

    row = [eval_method, config.MODEL.NAME, pp_str]

    # Ensure order matches header
    metric_order = ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
    for m in metric_order:
        if m in stats:
            val, std = stats[m]
            row.append(f'{val:.3f} ± {std:.3f}')
        else:
            row.append("")  # Empty if metric not calculated

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"   Saved result for {condition}")


def generate_plots(gt, pred, config, condition, dataset_name):
    """Generate Bland-Altman plots."""
    eval_method, use_bandpass, diff_flag, detrend_flag = condition

    if "BA" not in config.TEST.METRICS:
        return

    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    else:
        model_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = f"{model_root}_{dataset_name}"

    tags = f"{eval_method}_{'BP_On' if use_bandpass else 'BP_Off'}_{'Diff_On' if diff_flag else 'Diff_Off'}_{'Det_On' if detrend_flag else 'Det_Off'}"
    plot_title = f'{filename_id}_{tags}'

    compare = BlandAltman(gt, pred, config, averaged=True)
    compare.scatter_plot(
        x_label='GT PPG HR [bpm]', y_label='rPPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{plot_title}_Scatter', file_name=f'{plot_title}_Scatter.pdf')
    compare.difference_plot(
        x_label='Difference [bpm]', y_label='Average [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{plot_title}_BA', file_name=f'{plot_title}_BA.pdf')


# ---------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------

def calculate_metrics(predictions, labels, config):
    """
    Main entry point for calculating rPPG metrics.
    Orchestrates the loop over conditions, processing, and reporting.
    """
    dataset_name = config.TEST.DATA.DATASET
    model_name = config.MODEL.NAME
    csv_file = f'./results/result_{dataset_name}_{model_name}.csv'

    # 1. Initialize Report File
    initialize_csv(csv_file)

    print(f"====== DISPLAY METRICS ======")
    print(f"Dataset: {dataset_name} | Model: {model_name}")

    # 2. Iterate over All Conditions (Flattened Loop)
    conditions = get_test_conditions(config)  # Returns iterator of (method, bp, diff, detrend)

    for condition in conditions:
        print(f"\n-> Processing Condition: {condition}")

        # 3. Process Data
        gt_hr, pred_hr, snr = process_dataset_under_condition(predictions, labels, config, condition)

        # 4. Compute Statistics
        stats = compute_statistics(gt_hr, pred_hr, snr, config.TEST.METRICS)

        # 5. Save Results
        save_results_to_csv(csv_file, config, condition, stats)

        # 6. Generate Plots
        generate_plots(gt_hr, pred_hr, config, condition, dataset_name)