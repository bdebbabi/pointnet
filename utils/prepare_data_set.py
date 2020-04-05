# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from os.path import splitext, join
import numpy as np
from  tqdm import tqdm
import h5py
import pandas as pd
import open3d as o3d


# %%
# Select files
files = np.array(os.listdir('HeadPointsAndLabels'))
point_files = np.sort(files[[splitext(file)[1] == '.pts' for file in files]])
pid_files = np.sort(files[[splitext(file)[1] == '.seg' for file in files]])


# %%
# Get points
points = []
for file in tqdm(point_files): 
    points.append(pd.read_csv(join('HeadPointsAndLabels',file),sep =' ').to_numpy())
points= np.array(points)


# %%
# Get pid
pid = []
for file in tqdm(pid_files): 
    pid.append(pd.read_csv(join('HeadPointsAndLabels',file)).to_numpy().flatten())
pid = np.array(pid)


# %%
# Downsample points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[0])
# 2048 random points 
# pcd.points = o3d.utility.Vector3dVector(points[0][random.sample(range(points[0].shape[0]), 2048),:])

#pcd = pcd.uniform_down_sample(every_k_points=int(points[0].shape[0]/2048))
pcd = pcd.voxel_down_sample(voxel_size=10)
print(np.asarray(pcd.points).shape)
o3d.visualization.draw_geometries([pcd])


# %%
# Create h5 file
with h5py.File("ply_data_train.h5", "w") as train_file:
    train_file.create_dataset("data", data = points)
    train_file.create_dataset("pid", data = pid)
    train_file.create_dataset("label", data = np.array([0]))


# %%
# Read h5 file
with h5py.File("ply_data_train.h5", "r") as train_file:
    data = train_file['data']
    pid = train_file['pid']
    label = train_file['label']


# %%


