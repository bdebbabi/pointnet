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
from sklearn.model_selection import KFold


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
        new_class_cloud = np.asarray(
            class_cloud.remove_radius_outlier(nb_points=20, radius=11)[0].points)

        new_cloud = np.concatenate((new_cloud, new_class_cloud))
        new_labels.extend([label] * new_class_cloud.shape[0])

    X = new_cloud
    Y = np.array(new_labels)
    assert X.shape[0] == Y.shape[0]
    return X, Y


def random_downsample(X, Y, target_number=2048, keep_initial_density=False):
    """Performs a random down-sampling on the dataset"""
    classes, counts = np.unique(Y, return_counts=True)
    N_class = classes.shape[0]

    if keep_initial_density:
        total = Y.shape[0]
        strategy = {cls: int(count / total * target_number)
                    for cls, count in zip(classes, counts)}
    else:
        strategy = {cls: min(target_number // N_class, count)
                    for cls, count in zip(classes, counts)}
    # Pad using the class 0
    strategy[0] += target_number - np.sum(list(strategy.values()))

    rand = RandomUnderSampler(sampling_strategy=strategy, random_state=42)

    return rand.fit_resample(X, Y)


def get_normal(X, X_processed):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    normals = np.asarray(pcd.normals)

    indexes = [np.where((X == x).all(axis=1))[0][0] for x in X_processed]
    indexes = np.ix_(indexes, [0, 1, 2])

    normals = normals[indexes]

    return normals


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

    print(
        f">> n_classes = {N_classes}, n/cls = {N_per_class}, iter = {N_iter}")

    X, Y = shuffle_sample(X, Y)
    sample_partition = {cls: X[Y == cls] for cls in classes}
    seg_partition = {cls: Y[Y == cls] for cls in classes}
    X_new, Y_new = [], []
    for i in range(N_iter):
        first_class = classes[0]
        first_number = N_per_class + target_number - N_per_class * N_classes
        x = sample_partition[first_class][i *
                                          first_number:(i + 1) * first_number]
        y = seg_partition[first_class][i * first_number:(i + 1) * first_number]
        for cls in classes[1:]:
            x = np.concatenate(
                (x, sample_partition[cls][i * N_per_class:(i + 1) * N_per_class]))
            y = np.concatenate(
                (y, seg_partition[cls][i * N_per_class:(i + 1) * N_per_class]))
        X_new.append(x)
        Y_new.append(y)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new)

    return X_new, Y_new


def normalize_values(X):
    """Maps the values of X between -1.0 and 1.0"""
    min_max_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    return min_max_scaler.fit_transform(X)


def generate_data(X, Y, input_folder, target_number=2048, use_partitioning=False,
                  keep_initial_density=False, compute_normals=True, only_front_face=False):
    """Take a series of sample files
    and returns data """
    print(f">> Generating data")

    samples = []
    pids = []
    normals = []
    for x, y, in tqdm(zip(X, Y)):
        original_data, seg = load_sample(os.path.join(
            input_folder, x), os.path.join(input_folder, y))
        # Step 1: remove outliers
        data, seg = remove_outliers(original_data, seg)

        # Step 2: down-sample to target_number points
        if not use_partitioning:

            # Step 2.2: Take only the front facing part if required
            if only_front_face:
                norm_values = normalize_values(data)
                mask = norm_values[:, 1] < 0
                data = data[mask]
                seg = seg[mask]

            # Step 2.3: Downsampling
            data, seg = random_downsample(data, seg, target_number=target_number,
                                          keep_initial_density=keep_initial_density)

            assert data.shape[0] == target_number
            assert seg.shape[0] == target_number

            # Step 3: get normals
            if compute_normals:
                normal = get_normal(original_data, data)
                normals.append(normal)

            # Step 4: normalize between -1.0 and 1.0
            data = normalize_values(data)

            samples.append(data)
            pids.append(seg)
        else:
            partitions, segs = partition_sample(
                data, seg, target_number=target_number)

            assert partitions.shape[1] == target_number
            assert segs.shape[1] == target_number

            for partition in partitions:
                # Step 3: get normals
                if compute_normals:
                    normal = get_normal(original_data, partition)
                    normals.append(normal)

                # Step 4: normalize between -1.0 and 1.0
                partition = normalize_values(partition)
                samples.append(partition)

            pids.extend(segs)

    # There is only one class (Face)
    labels = np.zeros(len(samples))

    return np.array(samples), np.array(normals), labels, np.array(pids)


def save_files_to_h5(X, Y, N,  labels, output_file):
    """Turns data into an h5 file"""

    store = h5py.File(output_file, "w")
    store.create_dataset(
        'data', data=X,
        dtype='float32', compression='gzip', compression_opts=4
    )
    if compute_normals:
        store.create_dataset(
            'normal', data=N,
            dtype='float32', compression='gzip', compression_opts=4
        )
    store.create_dataset(
        'label', data=labels,
        dtype='uint8', compression='gzip', compression_opts=1
    )
    store.create_dataset(
        'pid', data=Y,
        dtype='uint8', compression='gzip', compression_opts=1
    )


# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./HeadPointsAndLabels',
                    help='Path to the dataset [default: ./HeadPointsAndLabels]')
parser.add_argument('--output', type=str, default='./DownSampledDataset',
                    help='Path the down-sampled dataset [default: ./part_seg/hdf5_data]')
parser.add_argument('--test_size', type=float, default=0.2,
                    help='Train/test ratio [default: 0.2]')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='Whether or not to shuffle the dataset when splitting [default: true]')
parser.add_argument('--num_points', type=int, default=2048,
                    help='The number of points to down-sample to [default: 2048]')
parser.add_argument('--use_partitioning', type=bool, default=False,
                    help='Whether or not use partitioning to down-sample samples [default: false]')
parser.add_argument('--keep_initial_density', type=bool, default=False,
                    help='Whether or not to down-sample using the same number of points for each classes [default: false]')
parser.add_argument('--compute_normals', type=bool, default=True,
                    help='Whether or not to compute the normals for the given dataset [default: true]')
parser.add_argument('--cross_validation', type=bool, default=False,
                    help='Whether or not to perform cross validation [default: false]')
parser.add_argument('--only_front_face', type=bool, default=False,
                    help='Whether or not to split the face in two, simulating a front facing scan like with the Kinect [default: False]')
FLAGS = parser.parse_args()

input_folder = FLAGS.input
output_folder = FLAGS.output
test_size = FLAGS.test_size
shuffle_dataset = FLAGS.shuffle
target_number = FLAGS.num_points
use_partitioning = FLAGS.use_partitioning
keep_initial_density = FLAGS.keep_initial_density
compute_normals = FLAGS.compute_normals
cross_validation = FLAGS.cross_validation
only_front_face = FLAGS.only_front_face

files = os.listdir(input_folder)
sfiles = np.sort([f for f in files if f.endswith('.pts')])
segfiles = np.sort([f for f in files if f.endswith('.seg')])

print(f">> Found {len(sfiles)} samples in {input_folder}")

if not os.path.exists(output_folder):
    print(f">> Creating folder {output_folder}")
    os.makedirs(output_folder)

samples, normals, labels, pids = generate_data(sfiles, segfiles, input_folder, target_number=target_number,
                                               use_partitioning=use_partitioning, keep_initial_density=keep_initial_density,
                                               compute_normals=compute_normals, only_front_face=only_front_face)

if cross_validation:
    n_splits = 4
    print(f">> Performing cross validation with {n_splits} splits")

    X_train, X_test, Y_train, Y_test, N_train, N_test = train_test_split(
        samples, pids, normals, shuffle=shuffle_dataset, test_size=test_size)
    kf = KFold(n_splits=n_splits)
    for index, split in enumerate(tqdm(kf.split(X_train))):
        print(f">> Split {index+1}")

        train_index, val_index = split[0], split[1]

        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        n_train, n_val = N_train[train_index], N_train[val_index]

        print(
            f">> Got {x_train.shape[0]} samples for training, {X_test.shape[0]} for testing, {x_val.shape[0]} for validation")

        save_files_to_h5(x_train, y_train, n_train, labels, os.path.join(
            output_folder, f'ply_data_train{index}.h5'))
        save_files_to_h5(x_val, y_val, n_val, labels, os.path.join(
            output_folder, f'ply_data_val{index}.h5'))

    save_files_to_h5(X_test, Y_test, N_test, labels,
                     os.path.join(output_folder, f'ply_data_test0.h5'))

    with open(os.path.join(output_folder, "train_hdf5_file_list.txt"), "w") as f:
        [f.write(f'ply_data_train{i}.h5\n') for i in range(5)]

    with open(os.path.join(output_folder, "val_hdf5_file_list.txt"), "w") as f:
        [f.write(f'ply_data_val{i}.h5\n') for i in range(5)]

    with open(os.path.join(output_folder, "test_hdf5_file_list.txt"), "w") as f:
        f.write('ply_data_test0.h5\n')

else:
    print(
        f">> Splitting train/test with ratio {100 * test_size}% for test dataset")

    X_train, X_test, Y_train, Y_test, N_train, N_test = train_test_split(
        samples, pids, normals, shuffle=shuffle_dataset, test_size=test_size)
    X_train, X_val, Y_train, Y_val, N_train, N_val = train_test_split(
        X_train, Y_train, N_train, test_size=0.25)  # 0.25 * 0.8 = 0.2

    print(
        f">> Got {X_train.shape[0]} samples for training, {X_test.shape[0]} for testing, {X_val.shape[0]} for validation")

    save_files_to_h5(X_train, Y_train, N_train, labels,
                     os.path.join(output_folder, f'ply_data_train0.h5'))
    save_files_to_h5(X_test, Y_test, N_test, labels,
                     os.path.join(output_folder, f'ply_data_test0.h5'))
    save_files_to_h5(X_val, Y_val, N_val, labels, os.path.join(
        output_folder, f'ply_data_val0.h5'))

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
