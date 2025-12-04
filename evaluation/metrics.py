import csv
import os
import numpy as np
import torch
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
from evaluation.post_process import calculate_metric_per_video  # 가정: 이 함수가 import 되어 있다고 전제


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""

    # 3. 파일명 포맷 변경: result_{DATASET}_{MODEL}.csv
    dataset_name = config.TEST.DATA.DATASET
    model_name = config.MODEL.NAME
    csv_file = f'./results/result_{dataset_name}_{model_name}.csv'

    # 2. CSV 헤더 순서 정의
    headers = ['EVALUATION_METHOD', 'MODEL', 'POST_PROCESSING', 'MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']

    # CSV 파일 초기화 (헤더 작성)
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    print(f"====== DISPLAY METRICS ======")
    print(f"Dataset: {dataset_name} | Model: {model_name}")

    # 1. config.INFERENCE.EVALUATION_METHOD 리스트 전체 순회
    # 예: config.INFERENCE.EVALUATION_METHOD = ["FFT", "peak detection"]
    for eval_method in config.INFERENCE.EVALUATION_METHOD:
        print(f"Processing Evaluation Method: {eval_method}")

        predict_hr_all = list()
        gt_hr_all = list()
        SNR_all = list()

        # Post Processing 설정 확인
        # Diff Flag 설정
        if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        else:
            raise ValueError("Unsupported label type in testing!")

        # Bandpass Filter 설정 (Config에 없으면 기본값 True로 가정)
        use_bandpass = True
        if hasattr(config.TEST, 'POST_PROCESS') and hasattr(config.TEST.POST_PROCESS, 'USE_BANDPASS'):
            use_bandpass = config.TEST.POST_PROCESS.USE_BANDPASS

        # 4. POST_PROCESSING 리스트 생성 (CSV 저장용)
        post_processing_list = [f"DiffNormalized={diff_flag_test}", f"Bandpass={use_bandpass}"]
        post_processing_str = " | ".join(post_processing_list)

        # 데이터 순회 및 HR 계산
        for index in tqdm(predictions.keys(), ncols=80, desc=f"Calculating {eval_method}"):
            prediction = _reform_data_from_dict(predictions[index])
            label = _reform_data_from_dict(labels[index])

            video_frame_size = prediction.shape[0]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(prediction), window_frame_size):
                pred_window = prediction[i:i + window_frame_size]
                label_window = label[i:i + window_frame_size]

                if len(pred_window) < 9:
                    continue

                # Method string normalization for calculate_metric_per_video
                if eval_method == "peak detection":
                    hr_method_arg = "Peak"
                elif eval_method == "FFT":
                    hr_method_arg = "FFT"
                else:
                    hr_method_arg = eval_method  # Fallback

                # 4. calculate_metric_per_video 호출 (diff_flag, use_bandpassfilter 전달)
                # 함수 시그니처가 (predictions, labels, fs, diff_flag, use_bandpass, hr_method) 형태라고 가정합니다.
                # rPPG-Toolbox 버전에 따라 인자 순서가 다를 수 있으니 확인 후 키워드 인자(keyword arguments) 사용을 권장합니다.
                gt_hr, pred_hr, snr, _ = calculate_metric_per_video(
                    pred_window,
                    label_window,
                    diff_flag=diff_flag_test,
                    fs=config.TEST.DATA.FS,
                    hr_method=hr_method_arg,
                    use_bandpassfilter=use_bandpass  # 요구사항 반영
                )

                gt_hr_all.append(gt_hr)
                predict_hr_all.append(pred_hr)
                SNR_all.append(snr)

        # 수집된 결과로 Metric 계산
        gt_hr_all = np.array(gt_hr_all)
        predict_hr_all = np.array(predict_hr_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_all)

        # CSV 저장을 위한 Row 데이터 시작
        # 컬럼 순서: EVALUATION_METHOD, MODEL, POST_PROCESSING
        result_row = [eval_method, model_name, post_processing_str]

        # Metric 계산 및 출력
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE = np.mean(np.abs(predict_hr_all - gt_hr_all))
                std_err = np.std(np.abs(predict_hr_all - gt_hr_all)) / np.sqrt(num_test_samples)
                result_row.append(f'{MAE:.3f} ± {std_err:.3f}')
                print(f"{eval_method} MAE: {MAE:.3f} +/- {std_err:.3f}")

            elif metric == "RMSE":
                RMSE = np.sqrt(np.mean(np.square(predict_hr_all - gt_hr_all)))
                std_err = np.std(np.square(predict_hr_all - gt_hr_all)) / np.sqrt(num_test_samples)
                result_row.append(f'{RMSE:.3f} ± {std_err:.3f}')
                print(f"{eval_method} RMSE: {RMSE:.3f} +/- {std_err:.3f}")

            elif metric == "MAPE":
                MAPE = np.mean(np.abs((predict_hr_all - gt_hr_all) / gt_hr_all)) * 100
                std_err = np.std(np.abs((predict_hr_all - gt_hr_all) / gt_hr_all)) / np.sqrt(num_test_samples) * 100
                result_row.append(f'{MAPE:.3f} ± {std_err:.3f}')
                print(f"{eval_method} MAPE: {MAPE:.3f} +/- {std_err:.3f}")

            elif metric == "Pearson":
                corr = np.corrcoef(predict_hr_all, gt_hr_all)
                pearson_val = corr[0][1]
                std_err = np.sqrt((1 - pearson_val ** 2) / (num_test_samples - 2))
                result_row.append(f'{pearson_val:.3f} ± {std_err:.3f}')
                print(f"{eval_method} Pearson: {pearson_val:.3f} +/- {std_err:.3f}")

            elif metric == "SNR":
                SNR_val = np.mean(SNR_all)
                std_err = np.std(SNR_all) / np.sqrt(num_test_samples)
                result_row.append(f'{SNR_val:.3f} ± {std_err:.3f}')
                print(f"{eval_method} SNR: {SNR_val:.3f} +/- {std_err:.3f} (dB)")

            elif "BA" in metric:
                # Bland-Altman Plot (CSV에는 저장하지 않고 Plotting만 수행)
                if config.TOOLBOX_MODE == 'train_and_test':
                    filename_id = config.TRAIN.MODEL_FILE_NAME
                else:
                    model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
                    filename_id = f"{model_file_root}_{dataset_name}"

                compare = BlandAltman(gt_hr_all, predict_hr_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_{eval_method}_BA_Scatter',
                    file_name=f'{filename_id}_{eval_method}_BA_Scatter.pdf')
                compare.difference_plot(
                    x_label='Difference [bpm]',
                    y_label='Average [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_{eval_method}_BA_Diff',
                    file_name=f'{filename_id}_{eval_method}_BA_Diff.pdf')

        # 파일에 결과 쓰기
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_row)