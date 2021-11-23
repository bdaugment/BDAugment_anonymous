import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class ModelNet40Cls(data.Dataset):
    def __init__(self, num_points=1024, transforms=None, train=True, download=True, keeprate=1):
        super().__init__()

        self.transforms = transforms
        self.current_dir = os.getcwd()
        self.base_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.folder = "data/modelnet40_ply_hdf5_2048"
        self.data_dir = os.path.join(self.base_dir, self.folder)
        self.url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        if download and not os.path.exists(self.data_dir):
            zipfile = os.path.join(BASE_DIR, os.path.basename(self.url))
            subprocess.check_call(
                shlex.split("curl {} -o {}".format(self.url, zipfile))
            )

            subprocess.check_call(
                shlex.split("unzip {} -d {}".format(zipfile, BASE_DIR))
            )

            subprocess.check_call(shlex.split("rm {}".format(zipfile)))

        self.train = train
        self.set_num_points(num_points)
        if self.train:
            self.files = _get_data_files(os.path.join(self.data_dir, "train_files.txt"))
        else:
            self.files = _get_data_files(os.path.join(self.data_dir, "test_files.txt"))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(self.data_dir, f.split('/')[-1]))

            point_list.append(points)
            label_list.append(labels)

        self.points = np.concatenate(point_list, 0)
        self.labels = np.concatenate(label_list, 0)

        if self.train and keeprate < 1:
            self.points = self.points[0:int(len(self.points) * keeprate)]
            self.labels = self.labels[0:int(len(self.labels) * keeprate)]
        del point_list
        del label_list

        self.num_class = np.max(self.labels) + 1
    def __getitem__(self, idx):

        pointcloud = self.points[idx][:self.num_points]
        label = self.labels[idx]
        if self.train:
            np.random.shuffle(pointcloud)
        return pointcloud, label



    def __len__(self):
        return self.points.shape[0]

    def set_num_points(self, pts):
        self.num_points = pts


