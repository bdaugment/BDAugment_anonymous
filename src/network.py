import torch
import torch.nn.functional as F
import torch.nn as nn



class model_aug(nn.Module):
    def __init__(self, num_basis, coef_scale, offset=True):
        super(model_aug, self).__init__()
        self.num_basis = num_basis
        self.coef_scale = coef_scale
        self.predict_offset = offset

        self.conv71 = torch.nn.Conv1d(64 + 3 + 3 + 1, 32, 1)
        self.conv72 = torch.nn.Conv1d(32, 16, 1)
        self.conv73 = torch.nn.Conv1d(16, 1, 1)
        self.bn71 = nn.BatchNorm1d(32)
        self.bn72 = nn.BatchNorm1d(16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, coef_feat, random_coef, basis, key_pts, w_pc):
        B, N, _ = w_pc.shape

        _, K, _ = key_pts.shape

        if self.predict_offset:
            coef_feat = torch.cat([coef_feat, random_coef.view(B, self.num_basis, 1, 1).expand(-1, -1, -1, K)], 2).view(
                B * self.num_basis, 70+1, K)
        else:
            coef_feat = torch.cat([coef_feat, random_coef], 2).view(
                B * self.num_basis, 70+1, K)
        coef_feat = F.relu(self.bn71(self.conv71(coef_feat)))
        coef_feat = F.relu(self.bn72(self.conv72(coef_feat)))
        coef_feat = self.conv73(coef_feat)
        coef_feat = torch.max(coef_feat, 2, keepdim=True)[0]

        if self.predict_offset:
            coef_feat = coef_feat.view(B, self.num_basis, 1) * self.coef_scale + random_coef.view(B, self.num_basis, 1)
        else:
            coef_feat = coef_feat.view(B, self.num_basis, 1)
        coef = coef_feat.permute(0, 2, 1)
        def_key_pts = key_pts + torch.bmm(coef, basis).view(B, K, 3)
        def_pc = torch.bmm(w_pc, def_key_pts)

        return def_key_pts, def_pc, coef



