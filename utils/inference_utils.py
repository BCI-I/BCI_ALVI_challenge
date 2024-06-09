# function for quanternions upsamplind 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np 
import time
from tqdm import tqdm

def normalize_quats(v):
    """
    [Time, n_bones, 4]
    """
    norm = np.linalg.norm(v, axis = -1, keepdims=True)
    return v / norm


def slide_window_inference(x_data, y_data, model, window_size, stride, ds_rate=8):
    """Apply model inference on data with sliding window approach.
    Inputs should have similar lenght.
    Args:
        x_data (np.ndarray): Input data for model prediction.
        y_data (np.ndarray): Target data for comparison.
        model: Model with an inference method and a device attribute.
        window_size (int): Size of the sliding window for prediction.
        stride (int): Step size between windows.

    Returns:
        tuple: Tuple containing two np.ndarrays for predictions and targets.
    """
    n_steps = (x_data.shape[0] - window_size) // stride + 1
    preds, targets = [], []
    
    stride_vr = int(stride//ds_rate)
    
    for i in range(n_steps):
        start, end = i * stride, i * stride + window_size
        x, y = x_data[start:end], y_data[start:end:ds_rate]
        y_hat = model.inference(x)
        preds.append(y_hat[-stride_vr:])
        targets.append(y[-stride_vr:])
    
    preds, targets = np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)
    max_len = min([len(preds), len(targets)])
    
    return preds[:max_len], targets[:max_len]


def get_predictions_and_targets(val_datasets, model, window_size, stride):
    all_preds, all_targets = [], []
    
    for val_sample in val_datasets:
        sample_preds, sample_targets = [], []
        
        for data in tqdm(val_sample.exps_data):
            data_myo, data_vr = data['data_myo'], data['data_angles']
            preds, targets = slide_window_inference(data_myo, data_vr, model, window_size, stride)
            sample_preds.append(preds)
            sample_targets.append(targets)
        
        sample_preds, sample_targets = np.concatenate(sample_preds, axis=0), np.concatenate(sample_targets, axis=0)
        all_preds.append(sample_preds)
        all_targets.append(sample_targets)
    
    return all_preds, all_targets

def calculate_angle_metrics(preds, targets):
    angle_degrees, corr_coefs = [], []
    
    for pred, target in zip(preds, targets):
        angle_degree = np.rad2deg(np.mean(np.abs(target - pred)))
        corr_coef = np.mean(np.cos(np.arccos(np.clip(np.sum(target * pred, axis=-1) / (np.linalg.norm(target, axis=-1) * np.linalg.norm(pred, axis=-1)), -1.0, 1.0))))
        
        angle_degrees.append(angle_degree)
        corr_coefs.append(corr_coef)
    return angle_degrees, corr_coefs



def get_angle_degree(y_hat, y_batch):
    """
    numpy format files.
    [batch, n_bones, 4, time]
    """
    time, n_bones, n_quat,  = y_hat.shape
    y_hat, y_batch = y_hat.reshape(-1, 4), y_batch.reshape(-1, 4)

    mult = np.sum(y_hat*y_batch, axis=-1)**2
    angle_degree = np.mean(np.arccos(np.clip((2*mult -1), -1, 1))/np.pi*180)
    return angle_degree


def smooth_two_values_slerp(prev, current, weight):
    """
    if weight = 1 -> no smoothing
    """
    rots = R.from_quat([prev, current])
    # rots = R.concatenate()
    # rots = prev.concatenate(current)
    slerp = Slerp([0, 1], rots)
    return slerp([weight]).as_quat()

def smooth_quats(data, weight, prev=None): 
    """
    data [time, 16, 4]
    weight - [0, 1]
    
    """
    n_bones = data.shape[1]
    
    if prev is None: 
        prev = data[0]
        
    for t in range(1, data.shape[0]): 
        new_value = []
        for prev_bone, curr_bone in zip(data[t-1], data[t]):
            new_value_bone = smooth_two_values_slerp(prev_bone, curr_bone, weight)
            new_value.append(new_value_bone)
        new_value = np.concatenate(new_value, 0)
        
        data[t] = new_value
        
    return data


def upsample_quats(data, orig_fps, expected_fps):
    """
    data - [time, 16, 4]
    orig_fps - 10 or 20. 
    desire_fps - 40. 
    
    """
    n_times, n_bones, _ = data.shape
    
    orig_times = np.linspace(0, 1, n_times)
    expected_times = np.linspace(0, 1, int(n_times * expected_fps/orig_fps))
    
    results = []
    for bone_idx in range(n_bones):
        
        rots = R.from_quat(data[:, bone_idx])
        slerp = Slerp(orig_times, rots)
        new_rots = slerp(expected_times).as_quat()
        results.append(new_rots)
    results = np.stack(results, 1)

    return results



def merge_two_videos_vertically(path1, path2, output_path):
    
    from moviepy.editor import VideoFileClip, clips_array
    
    clip1 = VideoFileClip(str(path1))
    clip2 = VideoFileClip(str(path2))
    
    clips = [[clip1],
             [clip2]]
 
    final_clip = clips_array(clips)
    final_clip.write_videofile(str(output_path), fps = clip1.fps)
    print(f'Video {output_path.stem} completed')



def smooth_ema(data, mult, prev=None):
    """
    [Time, ...]
    """
    if prev is None: 
        prev = data[0]
    
    
    for i in range(1, data.shape[0]): 
        data[i] = prev * mult + data[i] * (1 - mult)
        prev = data[i]
    return data

def normalize_quats(v):
    """
    [Time, n_bones, 4]
    """
    norm = np.linalg.norm(v, axis = -1, keepdims=True)
    return v / norm

def calculcate_latency(model, window_size, fps = 200, device = 'cuda'): 
    N = 30
    X = np.zeros([window_size, 8])
    
    start_time = time.time()
    for i in range(N): 
        model.inference(X, device) 
    res = (time.time() - start_time)/N
    lat = window_size/fps + res

    
    print(f'window size: {window_size} || device : {device}')
    print(f'General latency w/o overlap: {np.round(lat, 2)} sec')
    print(f'Model latency {np.round(res*1000, 3)} ms')

    return res