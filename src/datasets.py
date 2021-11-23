import torch
import numpy as np
import os
import random

current_dir = os.getcwd()
base_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
folder = "data/metahandels_raw"
data_dir = os.path.join(base_dir, folder)


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, phase="train", data_dir=data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.k = 20
        self.pc = np.load(os.path.join(
            self.data_dir, "original_raw_centroid.npy"), allow_pickle=True)
        self.file_list = np.load(os.path.join(
            self.data_dir, "file_list_raw_cen.npy"), allow_pickle=True)
        self.pc = self.pc[np.where(self.file_list==1)[0]]
        self.key_pts = np.load(os.path.join(
            self.data_dir, "key_points_available_raw_cen.npy"), allow_pickle=True)
        self.w_pc = np.load(os.path.join(
            self.data_dir, "pc_weights_available_raw_cen.npy"), allow_pickle=True)
        self.labels = np.load(os.path.join(self.data_dir, "label_available_raw.npy"))
        self.labels = np.expand_dims(self.labels, -1)
        #self.class_length = np.load(os.path.join(self.data_dir, "class_length_raw.npy"))
        self.metahandles = np.load(os.path.join(self.data_dir, "raw_unalign_centroid_4dir_1clloss_metahandles.npy"))
        self.coef_feat = np.load(os.path.join(self.data_dir, "raw_unalign_centroid_4dir_1clloss_coef_feat.npy"))
        assert len(self.pc) == len(self.labels)

    def __len__(self):
            return len(self.pc)

    def __getitem__(self, idx):
        src_id = idx
        src_pc = self.pc[src_id]
        key_pts = self.key_pts[src_id]
        w_pc = self.w_pc[src_id]
        metahandles = self.metahandles[src_id]
        coef_feat = self.coef_feat[src_id]
        label = self.labels[src_id]

        return src_pc, key_pts, w_pc, metahandles, coef_feat, label

