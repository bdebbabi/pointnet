import argparse
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


def normalize_values(X):
    """Maps the values of X between -1.0 and 1.0"""
    min_max_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    return min_max_scaler.fit_transform(X)


def save_files_to_h5(X, Y, input_folder, output_file, target_number=2048):
    """Take a series of sample files
    and turns it into a h5 file"""
    store = h5py.File(output_file, "w")
    samples = []
    pids = []

    # There is only one class (Face)
    labels = np.zeros(X.shape[0])
    for x, y, in tqdm(zip(X, Y)):
        data, seg = load_sample(os.path.join(input_folder, x), os.path.join(input_folder, y))
        # Step 1: remove outliers
        data, seg = remove_outliers(data, seg)
        # Step 2: down-sample to target_number points
        data, seg = random_downsample(data, seg, target_number)
        # Step 3: normalize between -1.0 and 1.0
        data = normalize_values(data)

        assert data.shape[0] == target_number
        assert seg.shape[0] == target_number

        samples.append(data)
        pids.append(seg)

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
FLAGS = parser.parse_args()

input_folder = FLAGS.input
output_folder = FLAGS.output
split_ratio = FLAGS.train_test_ratio
shuffle_dataset = FLAGS.shuffle
target_number = FLAGS.num_points

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
                 input_folder, os.path.join(output_folder, 'ply_data_train0.h5'), target_number=target_number)
save_files_to_h5(X_test, Y_test,
                 input_folder, os.path.join(output_folder, 'ply_data_test0.h5'), target_number=target_number)
save_files_to_h5(X_val, Y_val,
                 input_folder, os.path.join(output_folder, 'ply_data_val0.h5'), target_number=target_number)

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

with open(os.path.join(output_folder, 'all_object_categories.txt'), "w") as f:
    # We only have faces
    f.write("face\t0\n")

with open(os.path.join(output_folder, 'overallid_to_catid_partid.json'), "w") as f:
    catid_partid = [["0", i] for i in range(5)]
    json.dump(catid_partid, f)
