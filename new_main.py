""" The main function of rPPG deep learning pipeline (Integration Test Mode)."""

import argparse
import datetime
import random
import sys
import os
import glob
import numpy as np
import torch
import yaml
import copy

from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader

# =============================================================================
# Seeding
# =============================================================================
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

loader_mapping = {
    "UBFC-rPPG": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
    "PURE": data_loader.PURELoader.PURELoader,
    "SCAMPS": data_loader.SCAMPSLoader.SCAMPSLoader,
    "MMPD": data_loader.MMPDLoader.MMPDLoader,
    "BP4DPlus": data_loader.BP4DPlusLoader.BP4DPlusLoader,
    "BP4DPlusBigSmall": data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader,
    "UBFC-PHYS": data_loader.UBFCPHYSLoader.UBFCPHYSLoader,
    "iBVP": data_loader.iBVPLoader.iBVPLoader,
    "PhysDrive": data_loader.PhysDriveLoader.PhysDriveLoader,
    "vv100": data_loader.vv100Loader.vv100Loader,
    "LADH": data_loader.LADHLoader.LADHLoader,
    "SUMS": data_loader.SUMSLoader.SUMSLoader,
    "COHFACE": data_loader.COHFACELoader.COHFACELoader,
    "VIPL-HR": data_loader.VIPLHRLoader.VIPLHRLoader
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Helper Functions (Modularized Selectors)
# =============================================================================

def add_args(parser):
    """Adds arguments for parser."""
    # config_file은 이제 루프 안에서 동적으로 할당되지만,
    # get_config 함수가 args.config_file을 참조하므로 인자 정의는 유지합니다.
    # 기본값은 None으로 설정하거나 무시됩니다.
    parser.add_argument('--config_file', required=False,
                        default=None, type=str, help="The name of the config file.")
    return parser


def get_trainer_by_model(config, data_loader_dict):
    """Returns the initialized trainer instance based on config.MODEL.NAME"""
    model_name = config.MODEL.NAME

    trainer_mapping = {
        "Physnet": trainer.PhysnetTrainer.PhysnetTrainer,
        "iBVPNet": trainer.iBVPNetTrainer.iBVPNetTrainer,
        "FactorizePhys": trainer.FactorizePhysTrainer.FactorizePhysTrainer,
        "Tscan": trainer.TscanTrainer.TscanTrainer,
        "EfficientPhys": trainer.EfficientPhysTrainer.EfficientPhysTrainer,
        "DeepPhys": trainer.DeepPhysTrainer.DeepPhysTrainer,
        "BigSmall": trainer.BigSmallTrainer.BigSmallTrainer,
        "PhysFormer": trainer.PhysFormerTrainer.PhysFormerTrainer,
        "PhysMamba": trainer.PhysMambaTrainer.PhysMambaTrainer,
        "RhythmFormer": trainer.RhythmFormerTrainer.RhythmFormerTrainer
    }

    if model_name not in trainer_mapping:
        raise ValueError(f'Your Model ({model_name}) is Not Supported Yet!')

    return trainer_mapping[model_name](config, data_loader_dict)


def get_dataset_loader_class(dataset_name):
    """Returns the Dataset Loader Class based on dataset name"""

    if dataset_name not in loader_mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return loader_mapping[dataset_name]


def setup_dataloaders(config, mode="train_and_test"):
    """
    Initializes and returns the dataloaders based on the config.
    """

    # 헬퍼 함수: 로더 생성 로직
    def _create_loader(data_config, name, batch_size, shuffle, generator):
        if not (data_config.DATASET and data_config.DATA_PATH):
            return None

        loader_cls = get_dataset_loader_class(data_config.DATASET)
        dataset = loader_cls(
            name=name,
            data_path=data_config.DATA_PATH,
            config_data=data_config,
            device=config.DEVICE
        )
        return DataLoader(
            dataset=dataset,
            num_workers=16,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=generator
        )

    # 1. Unsupervised Method (단일 로더 반환)
    if mode == "unsupervised_method":
        return _create_loader(
            data_config=config.UNSUPERVISED.DATA,
            name="unsupervised",
            batch_size=1,
            shuffle=False,
            generator=general_generator
        )

    # 2. Train/Test Modes (Dict 반환)
    data_loader_dict = dict()

    # Train Loader
    if mode == "train_and_test":
        data_loader_dict['train'] = _create_loader(
            data_config=config.TRAIN.DATA,
            name="train",
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            generator=train_generator
        )

    # Valid Loader
    if mode == "train_and_test" and not config.TEST.USE_LAST_EPOCH:
        data_loader_dict['valid'] = _create_loader(
            data_config=config.VALID.DATA,
            name="valid",
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            generator=general_generator
        )

    # Test Loader
    if mode in ["train_and_test", "only_test", "loop_test"]:
        data_loader_dict['test'] = _create_loader(
            data_config=config.TEST.DATA,
            name="test",
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            generator=general_generator
        )

    return data_loader_dict


def run_single_model_process(config, model_name):
    print(f"\n{'-' * 20} Start Processing Model: {model_name} {'-' * 20}")

    # 모델별 Data Format 매핑 (User Provided)
    model_format_mapper = {
        'Physnet': ['NCDHW'],
        'Tscan': ['NDCHW'],
        'FactorizePhys': ['NCDHW'],
        'DeepPhys': ['NDCHW'],
        'PhysFormer': ['NCDHW'],
        'EfficientPhys': ['NDCHW'],
        'RhythmFormer': ['NDCHW'],
        'PhysMamba': ['NCDHW'],
        'iBVPNet': ['NCDHW']
    }

    current_config = config.clone()
    current_config.defrost()

    # 1. 모델 이름 설정
    current_config.MODEL.NAME = model_name

    # 2. DATA_FORMAT 자동 업데이트 로직
    if model_name in model_format_mapper:
        # 리스트에서 포맷 문자열 추출 (예: ['NCDHW'] -> 'NCDHW')
        new_format = model_format_mapper[model_name][0]

        # Train, Valid, Test 섹션에 모두 적용
        if 'TRAIN' in current_config and 'DATA' in current_config.TRAIN:
            current_config.TRAIN.DATA.DATA_FORMAT = new_format

        if 'VALID' in current_config and 'DATA' in current_config.VALID:
            current_config.VALID.DATA.DATA_FORMAT = new_format

        if 'TEST' in current_config and 'DATA' in current_config.TEST:
            current_config.TEST.DATA.DATA_FORMAT = new_format

        print(f"[Config Auto-Update] Set DATA_FORMAT to '{new_format}' for model '{model_name}'")
    else:
        print(f"[Config Warning] Model '{model_name}' not found in mapper. Using default format from config.")

    current_config.freeze()

    print('Current Configuration (Snippet):')
    print(f"Toolbox Mode: {current_config.TOOLBOX_MODE}")
    print(f"Dataset: {current_config.TRAIN.DATA.DATASET if current_config.TOOLBOX_MODE == 'train_and_test' else 'N/A'}")
    # 변경된 포맷 확인용 로그
    print(
        f"Data Format: {current_config.TRAIN.DATA.DATA_FORMAT if current_config.TOOLBOX_MODE == 'train_and_test' else 'N/A'}",
        end='\n\n')

    # Data Loader Setup
    dataloaders = setup_dataloaders(current_config, current_config.TOOLBOX_MODE)

    # Execution based on TOOLBOX_MODE
    if current_config.TOOLBOX_MODE == "train_and_test":
        model_trainer = get_trainer_by_model(current_config, dataloaders)
        model_trainer.train(dataloaders)
        model_trainer.test(dataloaders)

    elif current_config.TOOLBOX_MODE == "only_test":
        model_trainer = get_trainer_by_model(current_config, dataloaders)
        model_trainer.test(dataloaders)

    elif current_config.TOOLBOX_MODE == "unsupervised_method":
        if not current_config.UNSUPERVISED.METHOD:
            raise ValueError("Please set unsupervised method in yaml!")

        # dataloaders는 여기서 단일 DataLoader 객체
        for method in current_config.UNSUPERVISED.METHOD:
            print(f"Running Unsupervised Method: {method}")
            unsupervised_predict(current_config, {"unsupervised": dataloaders}, method)

    print(f"{'-' * 20} Finished Model: {model_name} {'-' * 20}\n")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Logging Setup
    log_filename = f'./logs/{str(datetime.datetime.now()).replace(":", "_")[:19]}_integration_log.txt'
    sys.stdout = open(log_filename, 'w')
    print(f"Logging to {log_filename}")

    # Parse initial arguments
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # Target Directory for configs
    config_dir = "configs/integration_configs"

    # configs/integration_configs 하위의 모든 .yaml 파일 검색
    if not os.path.exists(config_dir):
        print(f"Error: Directory '{config_dir}' does not exist.")
        sys.stdout.close()
        sys.exit(1)

    yaml_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in '{config_dir}'.")
        sys.stdout.close()
        sys.exit(0)

    print(f"Found {len(yaml_files)} configuration files to process: {yaml_files}\n")

    # =========================================================
    # Loop 1: Iterate over each YAML config file
    # =========================================================
    for yaml_file in yaml_files:
        print(f"\n{'=' * 40}")
        print(f"Processing Config File: {yaml_file}")
        print(f"{'=' * 40}")

        try:
            # 1. Update args.config_file to the current file
            args.config_file = yaml_file

            # 2. Load Configuration
            config = get_config(args)

            # 3. Check for Model List (or single string)
            # YAML 파일 하나에 모델 이름이 리스트로 있을 수도 있고, 하나만 있을 수도 있음
            model_names = config.MODEL.NAME
            if isinstance(model_names, str):
                model_names = [model_names]

            # unsupervised인 경우 MODEL.NAME이 의미가 없을 수도 있으나,
            # 보통 config 구조상 채워져 있거나 무시됨.
            # 만약 None이면 더미 리스트로 처리
            if model_names is None and config.TOOLBOX_MODE == "unsupervised_method":
                model_names = ["Unsupervised_Run"]

            # =========================================================
            # Loop 2: Iterate over models defined in the current YAML
            # =========================================================
            for model_name in model_names:
                try:
                    run_single_model_process(config, model_name)
                    # Clean GPU memory between runs
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"!! Error processing model '{model_name}' in file '{yaml_file}': {e}")
                    import traceback

                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"!! Critical Error loading config file '{yaml_file}': {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\nAll integration tests finished.")
    sys.stdout.close()
