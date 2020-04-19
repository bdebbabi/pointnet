import argparse
from random import shuffle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import open3d as o3d
import h5py
import pandas as pd


def load_sample(X_file, Y_file):
    """Load one sample from the .pts and the .seg files"""

    X = pd.read_csv(X_file, header=None, sep='\s+').to_numpy()
    seg = pd.read_csv(Y_file, header=None, sep='\s+').to_numpy()
    assert X.shape[0] == seg.shape[0]

    return X, seg


def remove_outliers(X, Y):
    """Removes class outliers from the point cloud"""
    new_cloud = np.empty([0, 3])
    new_labels = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)

    for label, class_cloud in enumerate(
            [pcd.select_down_sample(list(np.where(Y == label)[0])) for label in range(0, 5)]):
        new_class_cloud = np.asarray(class_cloud.remove_radius_outlier(nb_points=20, radius=11)[0].points)

        new_cloud = np.concatenate((new_cloud, new_class_cloud))
        new_labels.extend([label] * new_class_cloud.shape[0])

    X = new_cloud
    Y = np.array(new_labels)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def random_downsample(X, Y, target_number=2048):
    """Performs a random down-sampling on the dataset"""
    classes, counts = np.unique(Y, return_counts=True)
    N_class = classes.shape[0]

    strategy = {cls: min(target_number // N_class, count) for cls, count in zip(classes, counts)}
    # Pad using the class 0
    strategy[0] += target_number - np.sum(list(strategy.values()))

    rand = RandomUnderSampler(sampling_strategy=strategy, random_state=42)

    return rand.fit_resample(X, Y)


def shuffle_sample(X, Y):
    """Shuffles X and Y"""
    assert X.shape[0] == Y.shape[0]

    arr = list(range(X.shape[0]))
    shuffle(arr)

    return X[arr], Y[arr]


def partition_sample(X, Y, target_number=2048):
    """Take one N pts sample and return M T points samples"""

    classes, counts = np.unique(Y, return_counts=True)
    N_classes = classes.shape[0]
    N_per_class = target_number // N_classes
    N_iter = np.min(counts) // N_per_class

    print(f">> n_classes = {N_classes}, n/cls = {N_per_class}, iter = {N_iter}")

    X, Y = shuffle_sample(X, Y)
    sample_partition = {cls: X[Y == cls] for cls in classes}
    seg_partition = {cls: Y[Y == cls] for cls in classes}
    X_new, Y_new = [], []
    for i in range(N_iter):
        first_class = classes[0]
        first_number = N_per_class + target_number - N_per_class * N_classes
        x = sample_partition[first_class][i * first_number:(i+1) * first_number]
        y = seg_partition[first_class][i * first_number:(i+1) * first_number]
        for cls in classes[1:]:
            x = np.concatenate((x, sample_partition[cls][i * N_per_class:(i + 1) * N_per_class]))
            y = np.concatenate((y, seg_partition[cls][i * N_per_class:(i + 1) * N_per_class]))
        X_new.append(x)
        Y_new.append(y)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new)

    return X_new, Y_new


def normalize_values(X):
    """Maps the values of X between -1.0 and 1.0"""
    min_max_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    return min_max_scaler.fit_transform(X)


def save_files_to_h5(X, Y, input_folder, output_file_format, target_number=2048, use_partitioning=False):
    """Take a series of sample files
    and turns it into a h5 file"""
    samples = []
    pids = []

    for x, y, in tqdm(zip(X, Y)):
        data, seg = load_sample(os.path.join(input_folder, x), os.path.join(input_folder, y))
        # Step 1: remove outliers
        data, seg = remove_outliers(data, seg)
        # Step 2: normalize between -1.0 and 1.0
        data = normalize_values(data)
        # Step 3: down-sample to target_number points
        if not use_partitioning:
            data, seg = random_downsample(data, seg, target_number)

            assert data.shape[0] == target_number
            assert seg.shape[0] == target_number

            samples.append(data)
            pids.append(seg)
        else:
            partitions, segs = partition_sample(data, seg, target_number=target_number)

            print(x)
            print(partitions.shape)
            assert partitions.shape[1] == target_number
            assert segs.shape[1] == target_number

            samples.extend(partitions)
            pids.extend(segs)

    # There is only one class (Face)
    labels = np.zeros(len(samples))

    store = h5py.File(output_file_format.format("0"), "w")
    store.create_dataset(
        'data', data=samples,
        dtype='float32', compression='gzip', compression_opts=4
    )
    store.create_dataset(
        'label', data=labels,
        dtype='uint8', compression='gzip', compression_opts=1
    )
    store.create_dataset(
        'pid', data=pids,
        dtype='uint8', compression='gzip', compression_opts=1
    )


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./HeadPointsAndLabels',
                    help='Path to the dataset [default: ./HeadPointsAndLabels]')
parser.add_argument('--output', type=str, default='./DownSampledDataset',
                    help='Path the down-sampled dataset [default: ./part_seg/hdf5_data]')
parser.add_argument('--train_test_ratio', type=float, default=0.4, help='Train/test ratio [default: 0.4]')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='Whether or not to shuffle the dataset when splitting [default: true]')
parser.add_argument('--num_points', type=int, default=2048,
                    help='The number of points to down-sample to [default: 2048]')
parser.add_argument('--use_partitioning', type=bool, default=False,
                    help='Whether or not use partitioning to down-sample samples [default: false]')
FLAGS = parser.parse_args()

input_folder = FLAGS.input
output_folder = FLAGS.output
split_ratio = FLAGS.train_test_ratio
shuffle_dataset = FLAGS.shuffle
target_number = FLAGS.num_points
use_partitioning = FLAGS.use_partitioning

files = os.listdir(input_folder)
sfiles = np.sort([f for f in files if f.endswith('.pts')])
segfiles = np.sort([f for f in files if f.endswith('.seg')])

data = [(sfiles[i], segfiles[i]) for i in range(len(sfiles))]

print(f">> Found {len(sfiles)} samples in {input_folder}")
print(f">> Splitting train/test with ratio {100 * split_ratio}%")

X_train, X_test, Y_train, Y_test = train_test_split(sfiles, segfiles, shuffle=shuffle_dataset, test_size=split_ratio)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5)

print(f">> Got {X_train.shape[0]} samples for training, {X_test.shape[0]} for testing, {X_val.shape[0]} for validation")

if not os.path.exists(output_folder):
    print(f">> Creating folder {output_folder}")
    os.makedirs(output_folder)

save_files_to_h5(X_train, Y_train,
                 input_folder, os.path.join(output_folder, 'ply_data_train{}.h5'), target_number=target_number,
                 use_partitioning=use_partitioning)
save_files_to_h5(X_test, Y_test,
                 input_folder, os.path.join(output_folder, 'ply_data_test{}.h5'), target_number=target_number,
                 use_partitioning=use_partitioning)
save_files_to_h5(X_val, Y_val,
                 input_folder, os.path.join(output_folder, 'ply_data_val{}.h5'), target_number=target_number,
                 use_partitioning=use_partitioning)

with open(os.path.join(output_folder, "train_hdf5_file_list.txt"), "w") as f:
    f.write('ply_data_train0.h5\n')

with open(os.path.join(output_folder, "test_hdf5_file_list.txt"), "w") as f:
    f.write('ply_data_test0.h5\n')

with open(os.path.join(output_folder, "val_hdf5_file_list.txt"), "w") as f:
    f.write('ply_data_val0.h5\n')

with open(os.path.join(output_folder, 'part_color_mapping.json'), "w") as f:
    colors = [
        [
            0.65,
            0.95,
            0.05
        ],
        [
            0.35,
            0.05,
            0.35
        ],
        [
            0.65,
            0.35,
            0.65
        ],
        [
            0.95,
            0.95,
            0.65
        ],
        [
            0.95,
            0.65,
            0.05
        ]
    ]
    json.dump(colors, f)

with open(os.path.join(output_folder, 'catid_partid_to_overallid.json'), "w") as f:
    d = {'0': 0}
    json.dump(d, f)

with open(os.path.join(output_folder, 'all_object_categories.txt'), "w") as f:
    # We only have faces
    f.write("face\t0\n")

with open(os.path.join(output_folder, 'overallid_to_catid_partid.json'), "w") as f:
    catid_partid = [["0", i] for i in range(5)]
    json.dump(catid_partid, f)
