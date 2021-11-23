import torch
import time
import sklearn.metrics as metrics
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def weights_init1(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
        
def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


class PointcloudJitter_batch(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            jittered_data = pc.new(pc.size(1), 3).normal_(
                mean=0.0, std=self.std
            ).clamp_(-self.clip, self.clip)
            pc[i, :, 0:3] += jittered_data

        return pc

class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size()[0]
        dim = pc.size()[-1]

        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[dim])
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[dim])

            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda()) + torch.from_numpy(xyz2).float().cuda()

        return pc

def eval_one_epoch(model, loader):
    mean_correct = []
    test_pred = []
    test_true = []

    for j, data in enumerate(loader, 0):
        points, target = data
        #points = torch.from_numpy(points)
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.float().cuda(), target.cuda()
        classifier = model.eval()
        pred, _, _= classifier(points)
        pred_choice = pred.data.max(1)[1]

        test_true.append(target.cpu().numpy())
        test_pred.append(pred_choice.detach().cpu().numpy())

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)

    return test_acc

class PointcloudRotate_batch(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        #normals = points.size(1) > 3
        #if not normals:
        return torch.bmm(points, rotation_matrix.t().unsqueeze(0).expand(points.shape[0], -1, -1).float().cuda())
        # else:
        #     pc_xyz = points[:, 0:3]
        #     pc_normals = points[:, 3:]
        #     points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
        #     points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())
        #
        #     return points


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


def pc_normalize_torch(pc):
    #l = pc.shape[0]
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / m
    return pc