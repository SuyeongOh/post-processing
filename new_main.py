""" The main function of rPPG deep learning pipeline (Integration Test Mode - Dataset & Model Loop)."""

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =============================================================================
# Global Mappings
# =============================================================================

# 1. Model Name -> Trainer Class
Trainer_Map = {
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

# 2. Dataset Name -> Loader Class
Dataset_Map = {
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
    "VIPL-HR": data_loader.VIPLHRLoader.VIPLHRLoader,

    # Aliases
    "UBFC-phys": data_loader.UBFCPHYSLoader.UBFCPHYSLoader,
    "VitalVideos": data_loader.vv100Loader.vv100Loader
}

# 3. Model Name -> Data Format
Format_Map = {
    'Physnet': 'NCDHW',
    'Tscan': 'NDCHW',
    'FactorizePhys': 'NCDHW',
    'DeepPhys': 'NDCHW',
    'PhysFormer': 'NCDHW',
    'EfficientPhys': 'NDCHW',
    'RhythmFormer': 'NDCHW',
    'PhysMamba': 'NCDHW',
    'iBVPNet': 'NCDHW'
}

# =============================================================================
# Helper Functions
# =============================================================================

def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default=None, type=str, help="The name of the config file.")
    return parser

def get_trainer_by_model(config, data_loader_dict):
    """Returns the initialized trainer instance based on config.MODEL.NAME"""
    model_name = config.MODEL.NAME
    if model_name not in Trainer_Map:
        raise ValueError(f'Your Model ({model_name}) is Not Supported Yet!')
    return Trainer_Map[model_name](config, data_loader_dict)

def get_dataset_loader_class(dataset_name):
    """Returns the Dataset Loader Class based on dataset name"""
    if dataset_name not in Dataset_Map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return Dataset_Map[dataset_name]

def setup_dataloaders(config, mode="train_and_test"):
    """
    Initializes and returns the dataloaders based on the config.
    """
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

    if mode == "unsupervised_method":
        return _create_loader(
            data_config=config.UNSUPERVISED.DATA,
            name="unsupervised",
            batch_size=1,
            shuffle=False,
            generator=general_generator
        )

    data_loader_dict = dict()

    if mode == "train_and_test":
        data_loader_dict['train'] = _create_loader(
            data_config=config.TRAIN.DATA,
            name="train",
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=True,
            generator=train_generator
        )

    if mode == "train_and_test" and not config.TEST.USE_LAST_EPOCH:
        data_loader_dict['valid'] = _create_loader(
            data_config=config.VALID.DATA,
            name="valid",
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            generator=general_generator
        )

    if mode in ["train_and_test", "only_test", "loop_test"]:
        data_loader_dict['test'] = _create_loader(
            data_config=config.TEST.DATA,
            name="test",
            batch_size=config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            generator=general_generator
        )

    return data_loader_dict

def run_single_model_process(config, model_name, dataset_name):
    """
    Runs the pipeline for a single model name AND single dataset.
    """
    print(f"\n{'-'*20} Processing [Dataset: {dataset_name}] / [Model: {model_name}] {'-'*20}")

    current_config = config.clone()
    current_config.defrost()

    # 1. Update Model Name
    current_config.MODEL.NAME = model_name

    # 2. Update Dataset Name (Train/Valid/Test)
    if 'TRAIN' in current_config and 'DATA' in current_config.TRAIN:
        current_config.TRAIN.DATA.DATASET = dataset_name
    if 'VALID' in current_config and 'DATA' in current_config.VALID:
        current_config.VALID.DATA.DATASET = dataset_name
    if 'TEST' in current_config and 'DATA' in current_config.TEST:
        current_config.TEST.DATA.DATASET = dataset_name
    if 'UNSUPERVISED' in current_config and 'DATA' in current_config.UNSUPERVISED:
        current_config.UNSUPERVISED.DATA.DATASET = dataset_name

    # 3. Update Data Format (based on Model using Global Map)
    if model_name in Format_Map:
        new_format = Format_Map[model_name]
        if 'TRAIN' in current_config and 'DATA' in current_config.TRAIN:
            current_config.TRAIN.DATA.DATA_FORMAT = new_format
        if 'VALID' in current_config and 'DATA' in current_config.VALID:
            current_config.VALID.DATA.DATA_FORMAT = new_format
        if 'TEST' in current_config and 'DATA' in current_config.TEST:
            current_config.TEST.DATA.DATA_FORMAT = new_format
        print(f"  -> Set DATA_FORMAT to '{new_format}'")

    current_config.freeze()

    print('  -> Configuration Snapshot:')
    print(f"     Toolbox Mode: {current_config.TOOLBOX_MODE}")
    print(f"     Dataset: {current_config.TRAIN.DATA.DATASET if current_config.TOOLBOX_MODE == 'train_and_test' else 'N/A'}")
    print(f"     Model: {current_config.MODEL.NAME}")

    try:
        # Data Loader Setup
        dataloaders = setup_dataloaders(current_config, current_config.TOOLBOX_MODE)

        # Execution
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
            for method in current_config.UNSUPERVISED.METHOD:
                print(f"     Running Unsupervised Method: {method}")
                unsupervised_predict(current_config, {"unsupervised": dataloaders}, method)

    except Exception as e:
        print(f"\n!! Error in [Dataset: {dataset_name} / Model: {model_name}]: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'-'*20} Finished [Dataset: {dataset_name} / Model: {model_name}] {'-'*20}\n")

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Logging Setup
    log_filename = f'./logs/{str(datetime.datetime.now()).replace(":", "_")[:19]}_integration_log.txt'
    sys.stdout = open(log_filename, 'w')
    print(f"Logging to {log_filename}")

    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    config_dir = "configs/integration_configs"
    target_datasets = ['iBVP', 'UBFC-rPPG', 'UBFC-phys', 'PURE', 'SCAMPS', 'VitalVideos']

    if not os.path.exists(config_dir):
        print(f"Error: Directory '{config_dir}' does not exist.")
        sys.stdout.close()
        sys.exit(1)

    yaml_files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

    if not yaml_files:
        print(f"No YAML files found in '{config_dir}'.")
        sys.stdout.close()
        sys.exit(0)

    print(f"Found {len(yaml_files)} configuration files.")
    print(f"Target Datasets: {target_datasets}\n")

    # Loop 1: YAML Config Files
    for yaml_file in yaml_files:
        print(f"\n{'='*60}")
        print(f"Loading Config File: {yaml_file}")
        print(f"{'='*60}")

        try:
            args.config_file = yaml_file
            config = get_config(args)

            model_names = config.MODEL.NAME
            if isinstance(model_names, str):
                model_names = [model_names]

            if model_names is None and config.TOOLBOX_MODE == "unsupervised_method":
                model_names = ["Unsupervised_Run"]

            # Loop 2: Datasets
            for dataset_name in target_datasets:

                # Loop 3: Models
                for model_name in model_names:
                    run_single_model_process(config, model_name, dataset_name)
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"!! Critical Error loading config file '{yaml_file}': {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nAll integration tests finished.")
    sys.stdout.close()