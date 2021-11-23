import datasets
import torch
import numpy as np
import network
from utils import PointcloudScaleAndTranslate, eval_one_epoch, PointcloudJitter_batch, PointcloudRotate_batch
import argparse
import shutil
import logging
from tqdm import tqdm
import os
from pathlib import Path
import datetime
from losses import cls_loss
from pointnet_utils import PointNetCls
from ModelNet40Loader import ModelNet40Cls
import sklearn.metrics as metrics


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size during training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--epoch',  default=250, type=int, help='Epoch to run')
    parser.add_argument('--num_classes',  default=40, type=int, help='number of classes')
    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log path [default: None]')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint path [default: None]')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=5,
                        help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float,
                        default=0.5, help='Decay rate for lr decay')
    parser.add_argument('--gpu', type=str, default='1, 2', help='GPU to use')

    return parser.parse_args()



opt = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

experiment_dir = Path('./log/')
experiment_dir.mkdir(exist_ok=True)

if opt.log_dir is None:
    log_name = str(datetime.datetime.now().strftime(
        '%m-%d_%H-%M_')) + opt.env_name
    experiment_dir = experiment_dir.joinpath(log_name)
else:
    experiment_dir = experiment_dir.joinpath(opt.log_dir+'_train_baseline')
experiment_dir.mkdir(exist_ok=True)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = experiment_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)

shutil.copy('datasets.py', str(experiment_dir))
shutil.copy('losses.py', str(experiment_dir))
shutil.copy('network.py', str(experiment_dir))
shutil.copy('pointnet_utils.py', str(experiment_dir))
shutil.copy('train_baseline.py', str(experiment_dir))
shutil.copy('utils.py', str(experiment_dir))



def log_string(str):
    logger.info(str)
    print(str)


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s/log.txt' % log_dir)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_string('PARAMETER ...')
log_string(opt)



dataset_train = ModelNet40Cls(1024, transforms=None, train=True)
dataloader_train = torch.utils.data.DataLoader(dataset_train, drop_last=True,
                                               shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                               worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

dataset_cls_test = ModelNet40Cls(1024, transforms=None, train=False)
dataloader_cls_test = torch.utils.data.DataLoader(dataset_cls_test, drop_last=False,
                                                  shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                  worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

net = PointNetCls(opt.num_classes).cuda()
print(net)
net = torch.nn.DataParallel(net)


tot_curve = []
test_acc_curve = []
start_epoch = 0


optimizer = torch.optim.Adam(
    net.parameters(),
    lr=opt.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=opt.decay_rate)

scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=opt.lr_decay)
LEARNING_RATE_CLIP = 1e-7


best_acc = 0
current_epoch = 0
criterion = torch.nn.CrossEntropyLoss()
PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()  # conventional augmentation
PointcloudRotate_batch = PointcloudRotate_batch()
PointcloudJitter_batch = PointcloudJitter_batch()


for epoch in range(start_epoch, opt.epoch):

    log_string('Epoch %d (%d/%s):' % (current_epoch + 1, epoch + 1, opt.epoch))

    scheduler_c.step()



    net.train()
    buf = {"tot": [], "train_acc": [], "test_pred": [], "test_true": []}


    for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train), smoothing=0.9):
        points, label = data
        label = label[:, 0]
        points, label = points.cuda().float(), label.cuda().long()
        points = PointcloudScaleAndTranslate(points)
        points = PointcloudRotate_batch(points)
        points = PointcloudJitter_batch(points)
        points = points.transpose(2, 1).contiguous()
        pred_ori, tran_ori, global_feat_ori = net(points)
        optimizer.zero_grad()
        loss = cls_loss(pred_ori, label)
        loss.backward()
        optimizer.step()
        pred_choice = pred_ori.data.max(1)[1]
        buf["test_true"].append(label.cpu().numpy())
        buf["test_pred"].append(pred_choice.detach().cpu().numpy())
        buf['tot'].append(loss.cpu().item())

    tot_curve.append(np.mean(buf['tot']))
    accuracy = metrics.accuracy_score(np.concatenate(buf['test_true']), np.concatenate(buf['test_pred']))
    log_string("loss: %f, train_acc: %f" %
                (np.mean(buf['tot']),
                 accuracy))
    for b in buf:
        buf[b] = []

    test_acc = eval_one_epoch(net.eval(), dataloader_cls_test)
    log_string("test_acc: %f"  %
               (test_acc))
    test_acc_curve.append(test_acc)
    if test_acc > best_acc:
        best_acc = max(best_acc, test_acc)
        if best_acc > 0.88:
            savepath = str(checkpoints_dir) + '/best_model_%f.pth' % best_acc
            state = {
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tot_curve': tot_curve,
                'test_acc_curve': test_acc_curve
            }
            torch.save(state, savepath)
    log_string("best_acc: %f"  %
               (best_acc))


    savepath = str(checkpoints_dir) + '/last_model.pth'
    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tot_curve': tot_curve,
        'test_acc_curve': test_acc_curve
    }
    torch.save(state, savepath)
    current_epoch += 1
