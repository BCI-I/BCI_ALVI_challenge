import json
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import PillowWriter

from matplotlib import animation
import wandb
from pathlib import Path


json_path = 'big_global_data.json'
name_to_color = {'thumb': 'red',
                 'index': 'blue',
                 'middle': 'green',
                 'ring': 'orange',
                 'pinky': 'pink'}

# with open(json_path, 'r') as f:
#     json_dict = json.load(f)

def multiply_quant(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    return (r1 * r2).as_quat()
def apply_quat(q, v):
    return R.from_quat(q).apply(v)

def get_finger_vecs_custom(hand_data, finger_start_pose, name):
    e = np.array([1, 0, 0])


    quat1 = hand_dict['b_r_wrist']

    quat2 = hand_dict[f'b_r_{name}1']
    quat3 = hand_dict[f'b_r_{name}2']
    quat4  = hand_dict[f'b_r_{name}3']

    quat1_inv = R.from_quat(quat1).inv().as_quat()


    quats = [quat1, quat2, quat3, quat4]
    quats = [multiply_quant(quat1_inv, q) for q in quats]

    t_vecs = [apply_quat(q, e) for q in quats]

    prev = np.array(finger_start_pose[name])

    prev = apply_quat(quat1_inv, prev)


    t_vecs = [v * np.linalg.norm(prev)/2 for v in t_vecs]


    # get sequential summation of the vectors.
    t_vecs_new =[]

    t_vecs_new.append(prev)
    for v in t_vecs[1:]:
        prev = prev + v
        t_vecs_new.append(prev)

    t_vecs_new = np.stack(t_vecs_new)
    return t_vecs_new






def inverse_rotations(sample):


    ### normalisation if needed.
    quat_base = sample[0]
    quat_base_inv = R.from_quat(quat_base).inv().as_quat()

    quats_new = [multiply_quant(quat_base_inv, q) for q in sample]
    quats_new = np.stack(quats_new)

    return quats_new




class Hand:
    """
    Allows to get slice of Hand parameters.

    """
    name_to_idx = {"b_r_wrist": 0, "b_r_index1": 1,
                   "b_r_index2": 2, "b_r_index3": 3,
                   "b_r_middle1": 4, "b_r_middle2": 5,
                   "b_r_middle3": 6, "b_r_pinky1": 7,
                   "b_r_pinky2" : 8, "b_r_pinky3": 9,
                   "b_r_ring1": 10, "b_r_ring2": 11,
                   "b_r_ring3": 12, "b_r_thumb1": 13,
                   "b_r_thumb2": 14, "b_r_thumb3": 15}

    idx_to_name = {0: "b_r_wrist", 1: "b_r_index1",
               2: "b_r_index2", 3: "b_r_index3",
               4: "b_r_middle1", 5: "b_r_middle2",
               6: "b_r_middle3", 7: "b_r_pinky1",
               8: "b_r_pinky2", 9: "b_r_pinky3",
               10: "b_r_ring1", 11: "b_r_ring2",
               12: "b_r_ring3", 13: "b_r_thumb1",
               14: "b_r_thumb2", 15: "b_r_thumb3"}


    # normalize start point of hand

    finger_start_pose = {'thumb': np.array([ 1.79773151e-04, -9.56848596e-05,  1.40103373e-04]),
                         'index': np.array([ 4.79965970e-04, -3.64983564e-05,  1.17722357e-04]),
                         'middle': np.array([ 4.78224132e-04, -1.26683160e-05,  8.61489718e-06]),
                         'ring': np.array([ 4.43463102e-04, -3.26179546e-05, -8.73337173e-05]),
                         'pinky': np.array([ 3.89465811e-04, -6.83795585e-05, -1.75300101e-04])}


    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    name_to_color = {'thumb': 'red',
                 'index': 'blue',
                 'middle': 'green',
                 'ring': 'orange',
                 'pinky': 'pink'}
    # init method or constructor
    def __init__(self, data, hand_type='right'):
        """

        data - [frames, n_bones, 4] rotations
        """
        self.hand_type = hand_type

        self.azim, self.elev = -270, -220
        self.data = data
        self.all_points  = [] # N, 5, 4, 3
#         self.points = for

    def get_rotations(self, i):

        sample = self.data[i]


        ### normalisation if needed.
        quat_base = sample[self.name_to_idx['b_r_wrist']]
        quat_base_inv = R.from_quat(quat_base).inv().as_quat()

        quats_new = [multiply_quant(quat_base_inv, q) for q in sample]
        quats_new = np.stack(quats_new)

        return quats_new

    def get_frame_points(self, idx):
        points = []
        for name in self.finger_names:
            tmp = self.get_one_finger_points(idx, name)
            points.append(tmp)
        return points


    def get_one_finger_points(self,idx, name):

        sample = self.get_rotations(idx)

        quat1 = sample[self.name_to_idx['b_r_wrist']]
        quat2 = sample[self.name_to_idx[f'b_r_{name}1']]
        quat3 = sample[self.name_to_idx[f'b_r_{name}2']]
        quat4  = sample[self.name_to_idx[f'b_r_{name}3']]

        quats = [quat1, quat2, quat3, quat4]

        e = np.array([1, 0, 0])

        t_vecs = [apply_quat(q, e) for q in quats]
        prev = np.array(self.finger_start_pose[name])
        t_vecs = [v * np.linalg.norm(prev)/2 for v in t_vecs]

        # get sequential summation of the vectors.
        t_vecs_new =[]

        t_vecs_new.append(prev)
        for v in t_vecs[1:]:
            prev = prev + v
            t_vecs_new.append(prev)

        t_vecs_new = np.stack(t_vecs_new)
        return t_vecs_new

    def convert_all_frames_to_points(self):
        self.all_points = [self.get_frame_points(n) for n in range(len(self))]



    def __len__(self):
        return len(self.data)

    def init_3dplot(self):

        fig = plt.figure(figsize = (5, 5))
        # ax = Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(azim=self.azim, elev=self.elev)

        return fig, ax


    def visualize_one_frame(self, idx, fig=None):
        """
        Fig is not None for animation only.
        """

        animate = False if fig is None else True
        if animate:
            ax = fig.axes[0]
        else:
            fig, ax = self.init_3dplot()

        plt.axis('off')
        lim = 0.0008
        ax.set_xlim(-lim/4, 2*lim)
        ax.set_ylim(-lim/2,lim/2)
        ax.set_zlim(-lim/2,lim/2)


        points = self.get_frame_points(idx)

        ## wrist position
        ax.scatter(0, 0, 0, s = 100, c = 'black', marker = '*')

        for i, finger_points in enumerate(points):
            x, y, z = finger_points[:, 0], finger_points[:, 1], -finger_points[:, 2]

            ax.scatter(x, y, z, c = name_to_color[self.finger_names[i]], s = 150)
            ax.plot([0, *x], [0, *y], [0, *z],
                    color =name_to_color[self.finger_names[i]],
                    linewidth=4)


        if not animate:
            return fig



    def visualize_all_frames(self):
        fig, ax = self.init_3dplot()

        def animate(i):
            plt.cla()
            value_points = self.visualize_one_frame(i, fig)

        anim = animation.FuncAnimation(fig, animate, interval=1, frames =len(self), repeat=True, )

        return anim




def save_animation(anim, path, fps =  30):
    writergif = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writergif)
    plt.close()


def save_animation_mp4(anim, path, fps):
    FFwriter = animation.FFMpegWriter(fps=fps)
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}', end = "\r")
    anim.save(path, writer = FFwriter, progress_callback = progress_callback)


def visualize_val_moves(model, val_exps_data, epoch, device, window_size=256, pred_fps=200):
    old_fps = 200
    new_fps = 25
    step = old_fps//new_fps
    step_pred = pred_fps//pred_fps


    for n, raw_data in enumerate(val_exps_data):

        x, y = raw_data['data_myo'][:window_size], raw_data['data_vr'][:window_size]

        y_pred = model.inference(x, device)

        hand_gt = Hand(y[-256::step])
        hand_pred = Hand(y_pred[-256::step_pred])

        gt_path = f'{wandb.run.dir}/videos/{n}_move/true_sample_{epoch}.gif'
        pred_path = f'{wandb.run.dir}/videos/{n}_move/pred_sample_{epoch}.gif'
        Path(gt_path).parent.mkdir(parents=True, exist_ok=True)

        plt.close()
        ani_gt = hand_gt.visualize_all_frames()
        save_animation(ani_gt, gt_path, fps = new_fps)

        plt.close()
        ani_pred = hand_pred.visualize_all_frames()
        save_animation(ani_pred, pred_path, fps = new_fps)


        wandb.log({f"visualization/{n}_move": [wandb.Video(gt_path, fps=new_fps),
                                               wandb.Video(pred_path, fps=new_fps)]})
        del hand_gt
        del hand_pred


def merge_two_gifs(path1, path2, output_path, fps):
    import imageio
    import numpy as np

    #Create reader object for the gif
    gif1 = imageio.get_reader(path1)
    gif2 = imageio.get_reader(path2)

    #If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length())

    #Create writer object
    new_gif = imageio.get_writer(output_path)

    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()


def merge_two_videos(path1, path2, output_path):

    from moviepy.editor import VideoFileClip, clips_array

    clip1 = VideoFileClip(str(path1))
    clip2 = VideoFileClip(str(path2))

    clips = [[clip1, clip2]]

    final_clip = clips_array(clips)
    print('FPS', clip1.fps)
    final_clip.write_videofile(str(output_path), fps = clip1.fps)
    print(f'Video {output_path.stem} completed')






def visualize_and_save_anim(data, path, fps):
    target_hand = Hand(data)
    target_hand_anim = target_hand.visualize_all_frames()
    save_animation_mp4(target_hand_anim, path, fps=fps)
    plt.close()
    print(f'Video {path.stem} completed')


def visualize_and_save_anim_gifs(data, path, fps):
    target_hand = Hand(data)
    target_hand_anim = target_hand.visualize_all_frames()
    save_animation(target_hand_anim, path, fps=fps)
    plt.close()
    print(f'Video {path.stem} completed')


def merge_two_videos_vertically(path1, path2, output_path):

    from moviepy.editor import VideoFileClip, clips_array

    clip1 = VideoFileClip(str(path1))
    clip2 = VideoFileClip(str(path2))

    clips = [[clip1],
             [clip2]]

    final_clip = clips_array(clips)
    print('FPS', clip1.fps)
    final_clip.write_videofile(str(output_path), fps = int(clip1.fps))
    print(f'Video {output_path.stem} completed')
