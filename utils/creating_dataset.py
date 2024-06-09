import torch 
import numpy as np
from natsort import natsorted
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List
from .data_utils import VRHandMYODataset, adjust_delay_in_exp_data
from simple_parsing.helpers import Serializable

LEFT_TO_RIGHT_HAND = [6, 5, 4, 3, 2, 1, 0, 7]

def get_all_subjects_pathes(datasets):
    """
    Scan each dataset by .npz files.
    After that we get parent and build set top o that.
    So we get train and test pathes.
    """
    ALL_PATHS = []
    for d_path in datasets:
        data_dir = Path(d_path)
        file_names = list(data_dir.glob('**/*.npz'))
        paths = list(set([f.parent for f in file_names]))
        ALL_PATHS.extend(paths)
    ALL_PATHS = list(set(ALL_PATHS))
    return ALL_PATHS

def check_conditions(my_list, path):
    """
    Check whether path has folder with similar name as my list.
    So we can filter by train/test, left/right or together.
    """
    one_in_list = False
    for value in path.parts:
        if value in my_list:
            one_in_list = True
            break
    return one_in_list

def filter_by_condition(paths, condition):
    """
    Apply check condition for each path.
    Create new path list with "good" datasets.
    """
    FILTERED_PATHS = []
    for p in paths:
        if check_conditions(condition, p):
            FILTERED_PATHS.append(p)
    return FILTERED_PATHS

def get_train_val_pathes(config):
    """
    Config has to have ->
    config.datasets | config.human_type | config.hand_type | config.test_dataset_list
    Return:
    train and test pathes.
    """
    all_paths = get_all_subjects_pathes(config.datasets)

    filtered_paths = filter_by_condition(all_paths, config.human_type)
    filtered_paths = filter_by_condition(filtered_paths, config.hand_type)

    train_paths = filter_by_condition(filtered_paths, ['train'])
    test_paths = filter_by_condition(filtered_paths, ['test'])

    if config.test_dataset_list[0] != 'all':
        test_paths = filter_by_condition(test_paths, config.test_dataset_list)

    return sorted(train_paths), sorted(test_paths)

@dataclass
class DataConfig(Serializable):
    datasets: List[str]
    hand_type: List[str]
    human_type: List[str]
    test_dataset_list: List[str]

    use_angles: bool = True
    original_fps: int = 200  
    delay_ms: int = 0 
    start_crop_ms: int = 0 
    samples_per_epoch: int = 100_000
    window_size: int = 256
    down_sample_target: int = 8
    random_sampling: bool = True
    return_support_info: bool = False

def init_dataset(config: DataConfig, data_folder: Path, transform = None):
    
    """
    delay_ms - -40 it means emg[40:] and vr[:-40]
    dealy of emg compare with vr. vr changes and we'll see change in emg after 40 ms.
    """
    # Loop over files in data dir
    all_paths = sorted(data_folder.glob('*.npz'))
    all_paths = natsorted(all_paths)
    print(f'Number of moves: {len(all_paths)} | Dataset: {data_folder.parents[1].name}')

    exps_data = [dict(np.load(d)) for d in all_paths]

    # temporal alighnment
    n_crop_idxs = int(config.delay_ms/1000*config.original_fps)
    exps_data = [adjust_delay_in_exp_data(data, n_crop_idxs) for data in exps_data]

    # left hand -> right hand
    is_left_hand = check_conditions(['left'], data_folder)
    if is_left_hand:
        for i, data in enumerate(exps_data):
            exps_data[i]['data_myo'] = exps_data[i]['data_myo'][:, LEFT_TO_RIGHT_HAND]
        print('Reorder this dataset', data_folder.parents[1].name, is_left_hand)

    dataset = VRHandMYODataset(exps_data,
                            window_size=config.window_size,
                            random_sampling=config.random_sampling,
                            samples_per_epoch=config.samples_per_epoch,
                            return_support_info=config.return_support_info,
                            down_sample_target=config.down_sample_target,
                            use_angles = config.use_angles, 
                            transform = transform, 
                            path=data_folder.parents[1].name)
    return dataset

def get_datasets(config: DataConfig, transform=None, only_test=False):
    """
    Prepares and returns training and validation datasets based on the provided configuration.

    Args:
        config (DataConfig): Configuration data class containing dataset paths and parameters.
        transform (callable, optional): Transformation function to apply to the datasets. Train only.
        only_test (bool): If True, only the validation dataset is prepared and returned.

    Returns:
        tuple: A tuple containing the training dataset and validation dataset as `torch.utils.data.ConcatDataset` objects,
               unless `only_test` is True, in which case only the validation dataset is returned.
    """

    train_paths, val_paths = get_train_val_pathes(config)

    train_config = replace(config, samples_per_epoch=int(config.samples_per_epoch / len(train_paths)))               
    val_config = replace(config, random_sampling=False, samples_per_epoch=None)
    
    
    print('Getting val datasets')
    val_datasets = []
    for val_folder in val_paths:
        val_dataset = init_dataset(data_folder=val_folder,
                                   config=val_config,
                                   transform=None)
        val_datasets.append(val_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    
    if only_test:
        return val_dataset
    
    print('Getting train datasets')
    train_datasets = []
    for train_folder in train_paths:
        train_dataset = init_dataset(config=train_config,
                                     data_folder=train_folder, 
                                     transform=transform)
        if len(train_dataset)==0: 
            print('WWWWW: Problem with dataset', train_folder)
            break
        train_datasets.append(train_dataset)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    print(f"Number of trainining sessions: {len(train_dataset.datasets)}")
    print(f"Number of validation sessions: {len(val_dataset.datasets)}")
    print(f"Size of the input {train_dataset[0][0].shape} || Size of the output {train_dataset[0][1].shape}")

    return train_dataset, val_dataset



