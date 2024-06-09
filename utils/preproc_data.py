import numpy as np

from tqdm import tqdm
from pathlib import Path

from .data_utils import create_dataset_from_raw_files
from .quats_and_angles import get_angles

class DataConfig:
    original_fps = 200 
    delay_ms = 0
    start_crop_ms = 0
    window_size = 256

config = DataConfig()

def process_interim_datasets(ROOT_DIR, SAVE_DIR):
    """
    Datasets should be in train/test formats with npz files. 
    Script processes all datasets and save to SAVE_DIR.
    ROOT_DIR = Path('../../data/interim/dataset_v2_blocks/')
    SAVE_DIR = Path('../../data/processed/dataset_v2_blocks/')

    process_interim_datasets(ROOT_DIR, SAVE_DIR)
    """
    # get all paths to all folders inside above datasets 

    file_names = list(ROOT_DIR.glob('**/*.npz'))
    ALL_PATHS = list(set([f.parents[0] for f in file_names]))

    

    print('ALL_PATHS: ', ALL_PATHS)
    print('Number of paths: ', len(ALL_PATHS))

    # Preprocess all datasets for angles extraction.

    for path in ALL_PATHS:
        relative_path = path.relative_to(ROOT_DIR)

        dataset = create_dataset_from_raw_files(data_folder=path,
                                                  original_fps=config.original_fps,
                                                  delay_ms=config.delay_ms,
                                                  start_crop_ms=config.start_crop_ms,
                                                  window_size=config.window_size,
                                                  random_sampling=False,
                                                  transform=None)


        if len(dataset)==0: 
            print('WWWWW: Problem with dataset')
            break

        # go through each move and get angles and save.
        for idx, move in tqdm(enumerate(dataset.exps_data)):
            ts, myo, vr = move['myo_ts'], move['data_myo'], move['data_vr']
            angles = get_angles(vr)

            new_path = SAVE_DIR / relative_path.parent / Path('preproc_angles') / relative_path.name
            new_path.mkdir(parents=True, exist_ok=True)

            filename = f"{idx:04d}.npz"
            filepath = new_path / filename

            np.savez(filepath, data_myo=myo,
                     data_vr=vr, data_angles=angles, myo_ts=ts)
            
def get_fps(data):
   
    x = data['myo_ts'][~np.isnan(data['myo_ts'])]
    fps = (x[-1] - x[0]) / len(x)
    fps = 1/fps

    # fps = 1/np.nanmean(myo_ts[1:] - myo_ts[:-1])
    return fps

def get_files_with_small_fps(filenames, border):

    """
    Return filenames with fps << than border.
    """
    
    # calculate fps for all moves
    all_fps = []
    for file in filenames:
        data = np.load(file)
        fps = get_fps(data)
        all_fps.append(fps)
    all_fps = np.array(all_fps)
    
    args_bad_fps = np.argwhere(all_fps<border)
    bad_filenames = []
    for arg in args_bad_fps:
        bad_filenames.append(filenames[arg[0]])
        
    return bad_filenames


## TO DO. add argparse of the path and processing datasets. 