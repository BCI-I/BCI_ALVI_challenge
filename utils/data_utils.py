# Built-in
import logging
import random
import copy
from tqdm import tqdm
from typing import Union, NoReturn, Sequence, Optional, Dict, List, Tuple, Callable
from pathlib import Path
import time
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

# Extern
import numpy as np

from einops import rearrange


from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from simple_parsing import Serializable
# Custom

from .quats_and_angles import get_angles
logger = logging.getLogger(__name__)

# Table of contents 
# 1. Works with quats.
# 2. Filters dataset's folders
# 3. Interpolation functions
# 4. Functions for real time "smart" sampling
# 5. Torch datasets: Real-time and Pre-train.
# 6. (optional) -> augmentations


## Works with quats.
def multiply_quant(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()

def inverse_rotations(sample):
    ### normalisation if needed.
    quat_base = sample[0]
    quat_base_inv = R.from_quat(quat_base).inv().as_quat()

    quats_new = [multiply_quant(quat_base_inv, q) for q in sample]
    quats_new = np.stack(quats_new)

    return quats_new

def fix_sign_quats(data):
    """
    [Times, 16, 4]
    """
    n_times, n_bones, _ = data.shape

    data_tmp = data[:, :, -1].reshape(-1)
    data_sign = np.where(data_tmp<0, 1., -1.)
    data_sign = data_sign[..., None]

    data_new = data.reshape(-1, 4)
    data_new = data_new * data_sign
    data_new = data_new.reshape(n_times, n_bones, 4)
    return data_new

## Filters dataset's folders

def load_data_from_one_exp(file_path: Union[Path, str]) -> Dict['str', np.ndarray]:
    # np.load loads data from *.npz lazy so filedescriptor must be closed
    with np.load(file_path) as file:
        exp_data = dict(file)
    return exp_data

## Interpolation functions
def interpolate_quats_for_one_exp(data: Dict[str, np.ndarray],
                                  quat_interpolate_method: str = 'slerp') -> Dict[str, np.ndarray]:
    '''
    Inplace fill nan in quaternion_rotation positions (i.e. [:, :, 4:] slice) in data['data_vr']
    with interpolated quaternions based on existed values

    Args:
        data: dict with keys 'data_vr', 'data_myp', 'myo_ts', 'vr_ts' and corresonding np.ndarray values
        quat_interpolate_method (str): 'slerp' or 'nearest'(NotImplemented)

    Notes:
        (1) This function assume that vr_timestamps[0] and vr_timestamps[-1] is not np.nan

    Raises:
        ValueError: if myo_timestamps contains np.nan
    '''

    data = copy.deepcopy(data)

    data_vr: np.ndarray = data['data_vr']
    myo_timestamps: np.ndarray = data['myo_ts']
    vr_timestamps: np.ndarray = data['vr_ts']

    bones_amount = data_vr.shape[1]
    if np.isnan(myo_timestamps).any():
        raise ValueError('myo_timestamps contains np.nan')

    # find mask for not nan positions
    vr_timestamps_mask = ~np.isnan(vr_timestamps)
    masked_data_vr = data_vr[vr_timestamps_mask]
    masked_vr_timestamps = vr_timestamps[vr_timestamps_mask]

    # We will get interpoletion function for quats with vr_timestamps, but we would like to get quats for timestamps
    # for each myo_timestamps what require them to be inside [vr_ts[0], vr_ts[-1]]
    # so all data will be sliced over time to satisfy this requirement
    new_left_idx = np.argmax(myo_timestamps >= masked_vr_timestamps[0])
    new_right_idx = myo_timestamps.shape[0] - np.argmax(np.flip([myo_timestamps <= masked_vr_timestamps[-1]]))
    slice_by_n_values = (myo_timestamps.shape[0] - new_right_idx) + new_left_idx
    logger.debug(f'Slice myo_timestamps and all data from {new_left_idx} to {new_right_idx} by {slice_by_n_values} elements')

    if quat_interpolate_method == 'slerp':
        # iterate over each bones and append results to list 'interpolated_quats'
        # which then will be stacked over axis=1 to np.ndarray object
        interpolated_quats = []
        for bone_idx in range(bones_amount):
            # TODO move one iteration to single function
            _bone_quats = masked_data_vr[:, bone_idx, 4:8]
            _rotations = Rotation.from_quat(_bone_quats)
            slerp = Slerp(masked_vr_timestamps, _rotations)
            _quats = slerp(myo_timestamps[new_left_idx: new_right_idx]).as_quat()
            interpolated_quats.append(_quats)

        interpolated_quats = np.stack(interpolated_quats, axis=1)

    elif quat_interpolate_method == 'nearest':
        raise NotImplementedError

    # data is already deepcopy of original data passed to function so we dont need to make another copy
    # and can overwrite current values inside data
    data['data_vr'][new_left_idx: new_right_idx, :, 4:8] = interpolated_quats
    sliced_data = {k: v[new_left_idx: new_right_idx] for k, v in data.items()}

    return sliced_data

def strip_nans_for_one_exp(data: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], int, int]:
    '''
    Slice all array in dicts like arr[left_idx: right_idx + 1],
    where left_idx and right_idx are gotten from data['vr_ts']:

        [np.nan, np.nan, 2, 4, np.nan, 0.1, np.nan, 0.5, np.nan]
                         ^...........................^
                         |...........................|
                         .............................
                         left_idx....................right_idx

    '''
    vr_timestamps: np.ndarray = data['vr_ts']

    # find first and most latest positions with not np.nan values
    not_non_position = np.argwhere(~np.isnan(vr_timestamps))

    if len(not_non_position) ==0:
        return None, None, None

    left_idx = not_non_position[0][0]
    right_idx = not_non_position[-1][0]

    assert not np.isnan(vr_timestamps[left_idx])
    assert not np.isnan(vr_timestamps[right_idx])

    # arr[right_idx] is latest not nan item so slice must be before right_idx + 1 not inclusive
    stripped_data = {k: v[left_idx: right_idx + 1] for k, v in data.items()}
    return stripped_data, left_idx, right_idx

def calc_probas(iterable: Sequence[Sequence]) -> list:
    lens = []
    for child_iter in iterable:
        lens.append(len(child_iter))
    probas = np.array(lens) / sum(lens)
    return probas

def calc_stripped_len(sequence: Sequence, window_size: int) -> int:
    return int((len(sequence) // window_size) * window_size)

def find_first_nan_index(arr: np.ndarray) -> Union[int, None]:
    if arr.ndim != 1:
        raise ValueError(f'arr must be one dimentional but it have {arr.ndim}')
    nan_positions = np.where(np.isnan(arr))[0]
    if nan_positions.size == 0:
        return None
    else:
        return nan_positions[0]

def find_nonan_index_from_start(arr: np.ndarray) -> Union[int, None]:
    if arr.ndim != 1:
        raise ValueError(f'arr must be one dimentional but it have {arr.ndim}')
    nonan_positions = np.where(~np.isnan(arr))[0]
    if nonan_positions.size == 0:
        return None
    else:
        return nonan_positions[0]

def find_nonan_index_from_end(arr: np.ndarray) -> Union[int, None]:
    if arr.ndim != 1:
        raise ValueError(f'arr must be one dimentional but it have {arr.ndim}')
    nonan_positions = np.where(~np.isnan(arr))[0]
    if nonan_positions.size == 0:
        return None
    else:
        return nonan_positions[-1]

def find_indexes_for_strip_nans(arr: np.ndarray) -> Tuple[int, int]:
    not_non_position = np.argwhere(~np.isnan(arr))
    left_idx = not_non_position[0][0]
    right_idx = not_non_position[-1][0]
    return left_idx, right_idx

def get_dict_of_slerps(timestamps: np.ndarray,
                       hand_quaternions: np.ndarray) -> Dict[int, Slerp]:
    """
    timestamps(np.ndarray): timestamps array with shape [N_timestaps]
    hand_quaternions(np.ndarray): hand quaternions with shape [N_timestaps, 16, 4]
    """
    not_nan_positions = ~np.isnan(timestamps)
    timestamps_without_nans = timestamps[not_nan_positions]
    hand_quaternions_without_nans = hand_quaternions[not_nan_positions]

    dict_of_slerps = {}
    bones_amount = hand_quaternions_without_nans.shape[1]
    for bone_idx in range(bones_amount):
        _rotations = Rotation.from_quat(hand_quaternions_without_nans[:, bone_idx, :])
        dict_of_slerps[bone_idx] = Slerp(timestamps_without_nans, _rotations)
    return dict_of_slerps

def get_interpolated_hand(timestamps: Union[float, Sequence], dict_of_slerps: Dict[int, Slerp]) -> np.ndarray:
    '''
    '''
    interpolated_quats = []
    for bone_idx, slerp in dict_of_slerps.items():
        _quats = slerp(timestamps).as_quat()
        interpolated_quats.append(_quats)
    interpolated_quats = np.stack(interpolated_quats, axis=1)
    return interpolated_quats

## Functions for real time "smart" sampling
def petyaslava_distribution_function(i: int, d: int, max_idx: int) -> float:
    '''
    p(i) == 0
    sum([p(i) for i in range(0, max_idx)]) == 1

    p(i) ^   .....
         |  .    .
         | .     .
         |........
         ------------> i
             ^   ^
             |   |
     max_idx-d   max_idx
    '''
    H = 1 / (d + (max_idx - d + 1) / 2)
    if not (0 <= i <= max_idx):
        raise ValueError('i is out of distribution range')

    if i >= max_idx - d:
        return H
    else:
        return i * H / (max_idx - d)

def sashapetyaslava_distribution_function(i: int, d: int, max_idx: int) -> float:
    '''
    p(i) == 0
    sum([p(i) for i in range(0, max_idx)]) == 1

    p(i) ^   .....
         |   .   .
         |....   .
         |........
         ------------> i
             ^   ^
             |   |
     max_idx-d   max_idx
    '''
    h1 = 0.5 / (max_idx - d + 1)
    h2 = 0.5 / d
    if not (0 <= i <= max_idx):
        raise ValueError('i is out of distribution range')
    if i >= max_idx - d:
        return h2
    else:
        return h1

def get_sps_probs(d, max_idx):
    # if d >= max_idx:
    #     return np.ones(max_idx+1) / (max_idx+1)
    buf = np.ones(max_idx+1)
    h1 = 0.5 / (max_idx - d + 1)
    h2 = 0.5 / d
    buf[:max_idx - d + 1] = h1
    buf[max_idx - d + 1:] = h2
    return buf

def sample_from_petyaslava(d: int, max_idx: int) -> int:
    ws = range(0, max_idx + 1)
    probas = [petyaslava_distribution_function(i, d=d, max_idx=max_idx) for i in ws]
    return random.choices(ws, probas)[0]

def sample_from_sashapetyaslava(d: int, max_idx: int) -> int:
    # print(max_idx)
    if d >= max_idx + 1:
        return np.random.choice(max_idx + 1, replace=True, size=1)[0]
    probas = get_sps_probs(d=d, max_idx=max_idx)
    return np.random.choice(max_idx + 1, p = probas, replace=True, size=1)[0]


class VRHandMYODatasetRealtime(Dataset):
    # TODO
    # 1) Sampling from petya_slava in __getitem__ -> done
    # 2) ? randoms_sampling == False case         -> done, but need to rethink
    # 3) Test this dataset on old traning script (add update after each epoch) and describe dataset
    # 4) Test latency of dataset (add all files or add after each epoch) ##### to SLAVA
    # 5) samples_per_epoch defines frequency of append_new_data() as called after epoch
    #       add log one epoch time latency = 1/(frequincy of update) when number of epoch is ifinite
    # 6) save model weights each N epoch and log save latency
    #
    # Train loop:
    #   total_batches = 0
    #   epoch
    #       batch:
    #           total_batches += 1
    #           if total_batches % update_every_n_iteratorskk

    # Inference realtime:
    # if there are new file (check folder for new file (files[-2])) -> load new weights (next add timeout or something....)
    # next add with try except:
    #
    # 7) How to transfer saved model weights to another pc?
    # bash script to git push (first pc)
    # bash script to git pull (second pc)
    # ORRR via sockets


    # dont use dataloader


    # inf) refactoring

    MIRROR_QUATERNION_SLICE = (slice(None), slice(None), slice(4, 8, None))
    REAL_QUATERNION_SLICE = (slice(None), slice(None), slice(11, 15, None))
    DATA_KEYS = ['data_vr', 'data_myo', 'vr_ts', 'myo_ts', 'data_armband']

    def describe(self):
        print(f'{self.vr_out_data.shape=}')

    def __init__(self,
                 data_folder: Path,
                 vr_output_fps: int,
                 input_window_size: int,
                 samples_per_epoch: int,
                 petyaslava_p_out: int,
                 myo_input_fps: int = 200,
                 is_real_hand: bool = False,
                 random_sampling: str = 'last',
                 myo_transform: Optional[Callable] = None,
                 use_angles: bool = True,
                 debug_indexes: bool = False) -> NoReturn:

        self.probas = None #distribution for sampling on each epoch
        self.samples_per_epoch = samples_per_epoch  #defines frequency of dataset updating in training loop
        self.use_angles = use_angles
        self.myo_transform = myo_transform
        self.random_sampling = random_sampling
        self.input_window_size = input_window_size
        self.myo_input_fps = myo_input_fps
        self.vr_output_fps = vr_output_fps

        self.last_most_freq_out_indecies = petyaslava_p_out

        if self.myo_input_fps % self.vr_output_fps != 0:
            raise ValueError('')
        self.downsample_rate = self.myo_input_fps // self.vr_output_fps

        assert self.input_window_size % self.downsample_rate == 0
        self.output_window_size = self.input_window_size // self.downsample_rate

        # self.left_stride_to_left_border_intepolation_range = myo_input_fps // 2
        # self.left_stride_to_left_border_intepolation_range = myo_input_fps
        self.left_stride_to_left_border_intepolation_range = int(myo_input_fps * 1.5)

        self.is_real_hand = is_real_hand
        self.data_folder = data_folder
        self.appended_file_paths = []
        self.raw_data = {key: None for key in VRHandMYODatasetRealtime.DATA_KEYS}
        self.vr_out_ts = []
        self.vr_out_data = None
        self.vr_out_data_angles = None
        self.vr_out_data_wrist = None

        self.first_amount_vr_out_ts_skipped = 0
        self.append_counter = 0

        self.debug_indexes = debug_indexes
    def _out_index_to_input_index(self, index: int) -> int:
        input_index = (index + self.first_amount_vr_out_ts_skipped) * self.downsample_rate
        # print(input_index)
        return input_index

    def _check_new_files(self, filepath):
        if filepath:
            to_append = [filepath]
        else:
            to_append = []
            for p in self.data_folder.iterdir():
                if not (p in self.appended_file_paths):
                    to_append.append(p)
            to_append = sorted(to_append)[:-1]
            self.appended_file_paths += to_append
        return to_append

    def append_new_data(self, filepath: Union[None, Path] = None):
        # Data  appending
        # if filepath:
        #     to_append = [filepath]
        # else:
        #     to_append = []
        #     for p in self.data_folder.iterdir():
        #         if not (p in self.appended_file_paths):
        #             to_append.append(p)
        #     to_append = sorted(to_append)[:-1]
        #     self.appended_file_paths += to_append
        to_append = self._check_new_files(filepath)
        while not self.appended_file_paths:
            time.sleep(4)
            to_append = self._check_new_files(filepath)
            logger.warning('There are datasets but waiting new ones, waiting')

        print('files to append: ', to_append)
        if not to_append:
            logger.warning('There is no new files to append')
            return

        if self.raw_data['data_vr'] is None:
            data = {key: [] for key, value in self.raw_data.items() if key != 'data_armband'}
        else:
            data = {key: [value] for key, value in self.raw_data.items() if key != 'data_armband'}



        for i, p in enumerate(to_append):
            with np.load(p) as file:
                new_data = dict(file)

            start_slice_index = None
            if i == 0 and not self.append_counter:
                start_slice_index = find_first_nan_index(new_data['vr_ts'])
                if start_slice_index is not None:
                    start_slice_index += 1

            for key in data.keys():
                _to_append = new_data[key]
                # raw_data_myo first prerpocessing
                if key == 'data_myo':
                    # EMG preproc: normalize -> (-1, 1) range as audio.
                    _to_append = (_to_append + 128) / 255.
                    _to_append = 2 * _to_append - 1

                # raw_data_vr first prerpocessing before interpolation and etc
                if key == 'data_vr':
                    quaternion_slice = VRHandMYODatasetRealtime.REAL_QUATERNION_SLICE \
                        if self.is_real_hand else VRHandMYODatasetRealtime.MIRROR_QUATERNION_SLICE

                    # VR quats preproc:
                    _to_append = _to_append[quaternion_slice]
                    # concatenating armband data as one of bones (last)
                    _to_append = np.concatenate((_to_append,
                                                   new_data['data_armband'].reshape(len(new_data['data_armband']), 1, 4)),
                                                 axis = 1)
                    # print(_to_append.shape)
                    # print(new_data['data_armband'].shape)
            #         np.concatenate(
            # (self.raw_data['data_vr'],
            #  self.raw_data['data_armband'].reshape(len(self.raw_data['data_vr']), 1, 4)),
            # axis = 1 )
                    #

                # if key == 'data_wrist':
                # # no preproc needed appending data wrist

                if i == 0 and start_slice_index:
                    _to_append = _to_append[start_slice_index:, ...]

                data[key].append(_to_append)

        for key in data.keys():
            self.raw_data[key] = np.concatenate(data[key], axis=0)

        # concatenating armband data as one of bones (last)
        # self.raw_data['data_vr'] = np.concatenate(
        #     (self.raw_data['data_vr'],
        #      self.raw_data['data_armband'].reshape(len(self.raw_data['data_vr']), 1, 4)),
        #     axis = 1 )

        # Get boundaries for interpolation functions
        if self.vr_out_data is not None:
            anchor_index = self._out_index_to_input_index(len(self.vr_out_data) - 1)
            _left_idx = max(anchor_index - self.left_stride_to_left_border_intepolation_range, 0)
        else:
            _left_idx = 0

        # Get interpolation functions for each bone
        left_nonan_idx, right_nonan_idx = find_indexes_for_strip_nans(self.raw_data['vr_ts'][_left_idx:])
        left_nonan_idx += _left_idx
        right_nonan_idx += _left_idx
        left_interpolation_ts = self.raw_data['vr_ts'][left_nonan_idx]
        right_interpolation_ts = self.raw_data['vr_ts'][right_nonan_idx]
        dict_of_slerps = get_dict_of_slerps(timestamps=self.raw_data['vr_ts'][left_nonan_idx: right_nonan_idx+1],
                                            hand_quaternions=self.raw_data['data_vr'][left_nonan_idx: right_nonan_idx+1])

        print(f'{left_interpolation_ts=}, {right_interpolation_ts=}')
        downsample_slice = slice(None, None, self.downsample_rate)
        # vr_out_ts_old_len = len(self.vr_out_ts)

        vr_out_ts_old_len = len(self.vr_out_data) if self.vr_out_data is not None else 0 # ind in 25fps
        self.vr_out_ts = self.raw_data['myo_ts'][downsample_slice]   # ind in 25 fps
        new_elements_amount = len(self.vr_out_ts) - vr_out_ts_old_len  # this is len >=0 always
        new_vr_out_ts_elements = self.vr_out_ts[-new_elements_amount:] if new_elements_amount else []

        for vr_out_ts_element in new_vr_out_ts_elements:
            if np.isnan(vr_out_ts_element):
                raise ValueError('vr_out_ts_element is nan, which means myo_timestamps contains nan')

            if not (left_interpolation_ts <= vr_out_ts_element):
                logger.warning(f"New interpolation timestamps is out of range ({left_interpolation_ts=} <= {vr_out_ts_element=})")
                self.first_amount_vr_out_ts_skipped += 1
                if self.append_counter == 0:
                    print(f'YAAAABAAAAATTTT {self.append_counter=}')
                    continue
                else:
                    raise ValueError('Check self.left_stride_to_left_border_intepolation_range')

            if not (vr_out_ts_element <= right_interpolation_ts):
                logger.warning(f"Skip this ts ({vr_out_ts_element}) out of interpolation range")
                continue

            interpolated_hand = get_interpolated_hand(vr_out_ts_element, dict_of_slerps)
            interpolated_hand = rearrange(interpolated_hand, 'q b -> b q')
            # print('-----------')
            # print(interpolated_hand.shape)
            # print(interpolated_hand)
            # print('-----------')
            # Extract armband and wrist quaterions if armband quat in data
            have_armband = len(interpolated_hand) == 17
            if have_armband:
                q_armband = interpolated_hand[16, :].copy()
                q_wrist = interpolated_hand[0, :].copy()
                interpolated_hand = interpolated_hand[:-1, :]
                q_wrist_new = inverse_rotations([q_armband, q_wrist])[-1]

            interpolated_hand = inverse_rotations(interpolated_hand)  # wrist normalization
            interpolated_hand = rearrange(interpolated_hand, 'b q -> 1 b q')
            interpolated_hand = fix_sign_quats(interpolated_hand)  # fix ambiguity of quats because q = -q
            if self.vr_out_data is not None:
                self.vr_out_data = np.append(self.vr_out_data, interpolated_hand, axis=0)
            else:
                self.vr_out_data = interpolated_hand

            # Normalize wrist by armband quaternion and store it to self.vr_out_data_wrist
            if have_armband:
                q_wrist_new = rearrange(q_wrist_new, 'q -> 1 1 q')
                q_wrist_new = fix_sign_quats(q_wrist_new)
                q_wrist_new = q_wrist_new[0] # now shape is (1, 4)
                if self.vr_out_data_wrist is not None:
                    self.vr_out_data_wrist = np.append(self.vr_out_data_wrist, q_wrist_new, axis=0)
                else:
                    self.vr_out_data_wrist = q_wrist_new


            interpolated_hand_angles = get_angles(interpolated_hand)  # output shape is (1, 20), input shape is (1, 16, 4)
            if self.vr_out_data_angles is not None:
                self.vr_out_data_angles = np.append(self.vr_out_data_angles, interpolated_hand_angles, axis=0)
            else:
                self.vr_out_data_angles = interpolated_hand_angles

        self.append_counter += 1


    def __len__(self) -> int:
        vr_max_ind = len(self.vr_out_data)
        # myo_max_ind = self._out_index_to_input_index(vr_max_ind)
        return vr_max_ind
        # return self.samples_per_epoch
        # if self.random_sampling:
        #     return self.samples_per_epoch
        # else:
        #     raise NotImplementedError

    def get_slice_from_outind_to_end(self, vr_idx):
        # vr_idx should be len of dataset
        print()
        print('________________________')
        print('GET SLICE FROM DATASET')
        print(f'should be the same {self.first_amount_vr_out_ts_skipped=}')
        myo_idx = self._out_index_to_input_index(vr_idx)
        vr_len = len(self.vr_out_data)
        len_of_new_data_vr = vr_len - vr_idx - 1 #TODO think of -1
        len_of_new_data_myo = len_of_new_data_vr * self.downsample_rate
        print(f'{vr_idx=}')
        print(f'{myo_idx=}')
        print(f'{vr_len=}')


        if len_of_new_data_vr % self.output_window_size != 0:
            len_of_new_data_vr = int((len_of_new_data_vr // self.output_window_size) * self.output_window_size)
            len_of_new_data_myo = len_of_new_data_vr * self.downsample_rate
        print(f'{len_of_new_data_vr=}')
        print(f'{len_of_new_data_myo=}')


        assert len(self.vr_out_data_angles) >= vr_idx + len_of_new_data_vr
        if self.use_angles:
            vr_slice = self.vr_out_data_angles[vr_idx : vr_idx + len_of_new_data_vr]
            # vr_sample = rearrange(vr_sample, 't a -> a t')
        else:
            vr_slice = self.vr_out_data[vr_idx : vr_idx + len_of_new_data_vr]
            # vr_sample = rearrange(vr_sample, 't b q -> b q t')
        vr_slice = vr_slice.astype('float32')
        raw_myo_len = len(self.raw_data["data_myo"])
        print(f'{raw_myo_len=}')
        print(f'{myo_idx=}')
        print(f'{len_of_new_data_myo=}')

        # here is the problem myo could be -1
        assert raw_myo_len >= myo_idx + len_of_new_data_myo
        myo_slice = self.raw_data['data_myo'][myo_idx: myo_idx + len_of_new_data_myo]
        myo_slice = myo_slice.astype('float32')
        # myo_sample = rearrange(myo_sample, 't c -> c t')
        if self.myo_transform is not None:
            raise NotImplementedError
            # this thing maybe will not work
            # myo_sample = self.myo_transform(myo_sample)
        print('________________________')
        print()

        return myo_slice, vr_slice

    def _get_last_left_out_index(self):
        max_index = len(self.vr_out_data) - 1 - self.output_window_size
        # TODO delete
        #max_index += random.choices([0, 1], [0.9, .1], k=1)[0]
        assert max_index > 0
        min_index = 0
        return max_index

    def _get_random_ps_out_index(self):
        max_index = self._get_last_left_out_index()
        petyaslava_ind = sample_from_petyaslava(d=self.last_most_freq_out_indecies, max_idx=max_index)
        return petyaslava_ind

    def _get_random_sps_left_out_index(self):
        max_index = self._get_last_left_out_index()
        sashapetyaslava_ind = sample_from_sashapetyaslava(d=self.last_most_freq_out_indecies, max_idx=max_index)
        return sashapetyaslava_ind

    def _get_random_left_out_index_from_probas(self):
        max_idx = self._get_last_left_out_index()
        return np.random.choice(max_idx + 1, p = self.probas, replace=True, size=1)[0]


    def set_sampling_distribution(self, probas):
        max_index = self._get_last_left_out_index()
        print('CHANGING DISTRIBUTION')
        print(f'{probas.shape[0]=}, {max_index+1=}')
        assert probas.shape[0] == max_index+1
        self.random_sampling = 'custom'
        self.probas = probas


    def __getitem__(self, idx: int):
        if self.random_sampling == 'last':
            # always sample last affordable element
            output_random_index = self._get_last_left_out_index()
        elif  self.random_sampling == 'petyaslava':
            # sampling from petyaslava distribution
            output_random_index = self._get_random_ps_out_index()
        elif self.random_sampling == 'sashapetyaslava':
            # sampling from sashapetyaslava
            output_random_index = self._get_random_sps_left_out_index()
        elif self.random_sampling == 'custom':
            output_random_index = self._get_random_left_out_index_from_probas()
        else:
            raise ValueError(f'{self.random_sampling} distribution is not valid')



        if self.use_angles:
            vr_sample = self.vr_out_data_angles[output_random_index: output_random_index + self.output_window_size]
            vr_sample = rearrange(vr_sample, 't a -> a t')
        else:
            vr_sample = self.vr_out_data[output_random_index: output_random_index + self.output_window_size]
            vr_sample = rearrange(vr_sample, 't b q -> b q t')
        vr_sample = vr_sample.astype('float32')

        if self.vr_out_data_wrist is not None:
            vr_sample_wrist = self.vr_out_data_wrist[output_random_index: output_random_index + self.output_window_size]
            vr_sample_wrist = rearrange(vr_sample_wrist, 't a -> a t')
            vr_sample_wrist = vr_sample_wrist.astype('float32')
        else:
            vr_sample_wrist = None


        input_random_index = self._out_index_to_input_index(output_random_index)
        myo_sample = self.raw_data['data_myo'][input_random_index: input_random_index + self.input_window_size]
        myo_sample = myo_sample.astype('float32')
        myo_sample = rearrange(myo_sample, 't c -> c t')
        if self.myo_transform is not None:
            myo_sample = self.myo_transform(myo_sample)

        if self.debug_indexes:
            return myo_sample, vr_sample, output_random_index, input_random_index
        if vr_sample_wrist is None:
            return myo_sample, vr_sample
        else:
            return myo_sample, vr_sample, vr_sample_wrist

class VRHandMYODataset(Dataset):
    """
    A dataset class for handling VR hand movement data, particularly suited for
    working with MYO armband data in a virtual reality context.

    Parameters:
    - exps_data: List of dictionaries, each representing an experiment's data with keys mapping to np.ndarray.
    - window_size: The size of the sliding window to segment the data.
    - random_sampling: If True, samples windows randomly, otherwise sequentially.
    - samples_per_epoch: Number of samples per epoch, required if random_sampling is True.
    - return_support_info: If True, additional support info is returned with each sample.
    - transform: Optional transform to be applied on a sample.
    - down_sample_target: Factor by which the target data should be downsampled.
    - use_angles: If True, uses angle data as the target, otherwise uses VR data.
    """
    def __init__(self,
                 exps_data: List[Dict[str, np.ndarray]],
                 window_size: int,
                 random_sampling: bool = False,
                 samples_per_epoch: Optional = None,
                 return_support_info: bool = False,
                 transform=None,
                 down_sample_target=None,
                 use_angles=True, 
                 path = None) -> NoReturn:
        
        self.exps_data = exps_data
        self.path = path
        self.window_size = window_size
        self.random_sampling = random_sampling
        self.samples_per_epoch = samples_per_epoch
        self.return_support_info = return_support_info
        self.transform = transform
        self.down_sample_target=down_sample_target
        self.use_angles=use_angles

        if self.random_sampling:
            assert self.samples_per_epoch is not None, 'if random_sampling is True samples_per_epoch must be specified'

        # List of ints with strippred lens that shows which frames may and not get into any item
        self._stripped_lens = [calc_stripped_len(data['data_vr'], self.window_size) for data in exps_data]
        # Max numbers of different windows without intersections over all data
        self._items_per_stripped_exp = [_stripped_len // self.window_size for _stripped_len in self._stripped_lens]

        # Max left idx of window for each exp_data in self.exps_data
        self._max_left_idxs = [stripped_len - self.window_size for stripped_len in self._stripped_lens]
        # Probability to choose correspodint exp_data if random_sampling is passed
        self._exp_choose_probas = calc_probas(map(lambda x: x['data_vr'], self.exps_data))
        # print('Prob of different moves: ', self._exp_choose_probas)

    def __len__(self) -> int:
        if self.random_sampling:
            return self.samples_per_epoch

        assert sum(self._stripped_lens) % self.window_size == 0
        # Max numbers of different windows without intersections over all data
        max_items = sum(self._items_per_stripped_exp)
        return max_items

    def _window_left_idx_to_data_slice(self,
                                       exp_data: Dict[str, np.ndarray],
                                       idx: int) -> Dict[str, np.ndarray]:

        return {k: v[idx: idx+self.window_size] for k, v in exp_data.items()}

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], dict]:

        # Sample random window from random move.
        if idx >= len(self):
            raise IndexError

        if not self.random_sampling:
            running_lens_sum = 0
            for idx_of_exp, max_items in enumerate(self._items_per_stripped_exp):
                running_lens_sum += max_items
                if idx < running_lens_sum:
                    break

            window_idx = (idx - (running_lens_sum - max_items))
            window_left_idx = window_idx * self.window_size
            exp_data = self._window_left_idx_to_data_slice(self.exps_data[idx_of_exp], window_left_idx)

        else:
            idx_of_exp = random.choices(range(len(self._exp_choose_probas)), self._exp_choose_probas.tolist(), k=1)[0]
            window_left_idx = int(random.uniform(0, self._max_left_idxs[idx_of_exp]))
            exp_data = self._window_left_idx_to_data_slice(self.exps_data[idx_of_exp], window_left_idx)


        # INPUT data
        myo = exp_data['data_myo'].astype('float32')
        myo = rearrange(myo, 't c -> c t')

        if self.transform is not None:
            myo = self.transform(samples=myo, sample_rate=200)

        # TARGET data
        if self.use_angles:
            target = exp_data['data_angles'].astype('float32')
            target = rearrange(target, 't a -> a t')
        else:
            target = exp_data['data_vr'].astype('float32')
            target = rearrange(target, 't b q -> b q t')

        # downsampple target
        if self.down_sample_target is not None:
            target = target[..., ::self.down_sample_target]

        support_info = {
            'idx': idx,
            'idx_of_exp': idx_of_exp,
            'window_left_idx': window_left_idx,
            'len': len(exp_data)
        }

        # assert myo.shape[-1] == vr.shape[-1] == self.window_size, f'{myo.shape}, {vr.shape}, {support_info}'

        if self.return_support_info:
            return myo, target, support_info
        else:
            return myo, target



## Function for data processings.
def process_emg(emg):
    emg = (emg + 128) / 255.
    emg = 2 * emg - 1
    return emg

def process_quats(quats):
    quats = quats[:, :, 4:8]  # 4:8 is for mirror hand
    quats = np.stack([inverse_rotations(r) for r in quats])  # wrist normalization
    quats = fix_sign_quats(quats)  # fix ambiguity of quats because q = -q
    return quats

def crop_beginning_data(data, start_crop_idxs):
    if start_crop_idxs != 0:
        logger.warning(f'We assume start_crop_ms == 0 but start_crop_ms == {start_crop_ms}')
    
    data['data_vr'] = data['data_vr'][start_crop_idxs:]
    data['data_myo'] = data['data_myo'][start_crop_idxs:]
    data['myo_ts'] = data['myo_ts'][start_crop_idxs:]
    return data 

def adjust_delay_in_exp_data(exp_data, n_crop_idxs):
    # Calculate the number of indices to crop and its absolute value for slicing
    if n_crop_idxs == 0:
        return exp_data

    # Keys in each data dictionary that require special handling
    input_keys = ['myo_ts', 'data_myo']
    for key, value in exp_data.items():
        if n_crop_idxs > 0:
            # For positive delay, adjust data based on whether the key is in 'input_keys'
            exp_data[key] = value[:-n_crop_idxs] if key in input_keys else value[n_crop_idxs:]
        elif n_crop_idxs < 0:
            # For negative delay, use absolute value for slicing
            exp_data[key] = value[-n_crop_idxs:] if key in input_keys else value[:n_crop_idxs]
    return exp_data

def process_raw_data_file(path, original_fps, start_crop_ms, delay_ms):
    
    data = dict(np.load(path))
    data, left_strip_idx, right_strip_idx = strip_nans_for_one_exp(data)

    if data is None:
        print('No VR for this file:', path)
        return None

    data = interpolate_quats_for_one_exp(data, quat_interpolate_method='slerp')

    data['data_myo'] = process_emg(data['data_myo'])
    data['data_vr'] = process_quats(data['data_vr'])
    
    start_crop_idxs = int(start_crop_ms/1000*original_fps)
    delay_idxs = int(delay_ms / 1000 * original_fps)

    data = crop_beginning_data(data, start_crop_idxs)
    data = adjust_delay_in_exp_data(data, delay_idxs)

    assert data['data_vr'].shape[0] == data['data_myo'].shape[0], \
        f'lens of data_vr and data_myo are different {data["data_vr"].shape} !=  {data["data_myo"].shape}'
    
    return data 

def create_dataset_from_raw_files(data_folder, original_fps, start_crop_ms, delay_ms,
                                window_size, random_sampling, samples_per_epoch, 
                                return_support_info, transform, down_sample_target):
    # Loop over files in data dir
    all_paths = sorted(data_folder.glob('*.npz'))
    all_paths = natsorted(all_paths)

    print(f'Number of moves: {len(all_paths)} | Dataset: {data_folder.parents[1].name}')

    exps_data = []
    for one_exp_data_path in tqdm(all_paths):
        data = process_raw_data_file(one_exp_data_path, original_fps, start_crop_ms, delay_ms)
        exps_data.append(data)

    dataset = VRHandMYODataset(exps_data,
                                window_size=window_size,
                                random_sampling=random_sampling,
                                samples_per_epoch=samples_per_epoch,
                                return_support_info=return_support_info,
                                transform = transform,
                                down_sample_target=down_sample_target)

    print(f'Total len: {len(dataset)}')  # max numbers of different windows over
    return dataset