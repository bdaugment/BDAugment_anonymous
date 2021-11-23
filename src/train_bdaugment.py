import datasets
import torch
import numpy as np
import network
from utils import weights_init, PointcloudScaleAndTranslate, \
    eval_one_epoch, PointcloudJitter_batch, PointcloudRotate_batch
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
from losses import chamfer_distance


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size during training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for data loader')
    parser.add_argument('--epoch',  default=250, type=int, help='Epoch to run')
    parser.add_argument('--num_classes',  default=40, type=int, help='number of classes')
    parser.add_argument('--learning_rate', default=0.001,
                        type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=5,
                        help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float,
                        default=0.5, help='Decay rate for lr decay')
    parser.add_argument('--gpu', type=str, default='0,1,2', help='GPU to use')
    parser.add_argument('--display', type=int, default=50,
                        help='number of iteration per display')
    parser.add_argument('--env_name', type=str,
                        default='train_cls_basis', help='enviroment name of visdom')
    parser.add_argument('--num_basis', type=int, default=15,
                        help='number of basis vectors')
    parser.add_argument('--adv_weight', type=float,
                        default=0.01, help='weight for adversarial loss')
    parser.add_argument('--cd_weight', type=float,
                        default=0.1, help='weight for adversarial loss')
    parser.add_argument('--sym_weight', type=float,
                        default=0.1, help='weight for symmetry loss')
    parser.add_argument('--std', type=float,
                        default=0.5, help='standard deviation for random coef')
    parser.add_argument('--coef_scale', type=float,
                        default=0.1, help='scale for generated coef')
    parser.add_argument('--inner',  default=10, type=int, help='inner iterations to run')
    parser.add_argument('--outer',  default=15, type=int, help='outer iterations to run')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--offset', type=int, default=1,
                        help='predict offset')
    parser.add_argument('--include_original', type=int, default=1,
                        help='include original data in the loop')
    parser.add_argument('--finetune', type=int, default=1,
                        help='finetuning')
    parser.add_argument('--finetune_lr_scale', type=float,
                        default=0.1, help='scale for generated coef')
    return parser.parse_args()



opt = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

experiment_dir = Path('./log/')
experiment_dir.mkdir(exist_ok=True)

if opt.log_dir is None:
    log_name = str(datetime.datetime.now().strftime(
        '%m-%d_%H-%M_')) + opt.env_name
    experiment_dir = experiment_dir.joinpath(log_name)
else:
    experiment_dir = experiment_dir.joinpath(opt.log_dir)
experiment_dir.mkdir(exist_ok=True)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = experiment_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)

#if opt.log_dir is None:
shutil.copy('datasets.py', str(experiment_dir))
shutil.copy('losses.py', str(experiment_dir))
shutil.copy('network.py', str(experiment_dir))
shutil.copy('pointnet_utils.py', str(experiment_dir))
shutil.copy('train_bdaugment.py', str(experiment_dir))
shutil.copy('utils.py', str(experiment_dir))


def log_string(str):
    logger.info(str)
    print(str)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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


dataset_train_aug = datasets.AugDataset("train")
dataloader_train_aug = torch.utils.data.DataLoader(dataset_train_aug, drop_last=True,
                                                   shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                   worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

dataset_cls_train = ModelNet40Cls(1024, transforms=None, train=True)
dataloader_cls_train = torch.utils.data.DataLoader(dataset_cls_train, drop_last=True,
                                                   shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                   worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

dataset_cls_test = ModelNet40Cls(1024, transforms=None, train=False)
dataloader_cls_test = torch.utils.data.DataLoader(dataset_cls_test, drop_last=False,
                                                  shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                  worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

net_c = PointNetCls(opt.num_classes).cuda()
print(net_c)
net_c = torch.nn.DataParallel(net_c)

net_a = network.model_aug(opt.num_basis, opt.coef_scale, bool(opt.offset)).cuda()
print(net_a)
net_a = torch.nn.DataParallel(net_a)

try:
    checkpoint = torch.load(str(experiment_dir) +
                            '/checkpoints/last_model.pth')
    start_epoch = checkpoint['epoch'] // opt.inner + 1
    # test_acc_curve = []
    # tot_curve = []
    net_c.load_state_dict(checkpoint['model_state_dict_c'])
    net_a.load_state_dict(checkpoint['model_state_dict_a'])
    log_string('Use pretrain model')
except:
    log_string('No existing model, starting training from scratch...')
    start_epoch = 0
    net_c = net_c.apply(weights_init)
    net_a = net_a.apply(weights_init)


optimizer_c = torch.optim.Adam(
    net_c.parameters(),
    lr=opt.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=opt.decay_rate)

optimizer_a = torch.optim.Adam(
    net_a.parameters(),
    lr=opt.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=opt.decay_rate)

scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=20, gamma=opt.lr_decay)
scheduler_a = torch.optim.lr_scheduler.StepLR(optimizer_a, step_size=10, gamma=opt.lr_decay)


best_acc = 0
epoch = 0
criterion = torch.nn.CrossEntropyLoss()
PointcloudScaleAndTranslate = PointcloudScaleAndTranslate()  # conventional augmentation
PointcloudRotate_batch = PointcloudRotate_batch()
PointcloudJitter_batch = PointcloudJitter_batch()

inner_iter = opt.inner
MSE_criterion = torch.nn.MSELoss()
net_a.train()
net_c.train()



for outer_iter in range(start_epoch, opt.outer):


    # training net_c with clean and augmented samples
    for cls_i in range(inner_iter):
        log_string('Outer iteration %d (%d/%s):' % (outer_iter + 1, outer_iter + 1, opt.outer))
        epoch += 2
        net_c.train()
        net_a.eval()
        train_loss_clean = []
        train_loss_aug_cls = []
        train_pred = []
        train_true = []

        for i, data in tqdm(enumerate(dataloader_train_aug), total=len(dataloader_train_aug), smoothing=0.9):
            src_pc, key_pts, w_pc, metahandles, coef_feat, label = data
            src_pc, key_pts, w_pc, metahandles, coef_feat, label = src_pc.cuda().float(), \
                                                                   key_pts.cuda().float(), \
                                                                   w_pc.cuda().float(), \
                                                                   metahandles.cuda().float(), \
                                                                   coef_feat.cuda().float(), \
                                                                   label.cuda().squeeze().to(torch.int64)
            metahandles = metahandles.view(-1, 15, 150)
            B, N, K = w_pc.shape

            if opt.offset:
                random_coef = torch.normal(0, opt.std, size=(B, opt.num_basis, 1, 1)).cuda()
            else:
                random_coef = torch.normal(0, opt.std, size=(B, opt.num_basis, 1, K)).cuda()

            def_key_pts, def_pc, coef = net_a(coef_feat, random_coef, metahandles, key_pts, w_pc)
            recover_pc = torch.bmm(w_pc, key_pts)


            def_pc_input = def_pc.clone()
            def_pc_input[:, :, 1] = def_pc[:, :, 2].clone()
            def_pc_input[:, :, 2] = def_pc[:, :, 1].clone()
            def_pc_input = def_pc_input.transpose(2, 1).contiguous()

            pred_def1, tran_def1, global_feat_def1 = net_c(def_pc_input)

            recover_pc_input = recover_pc.clone()
            recover_pc_input[:, :, 1] = recover_pc[:, :, 2].clone()
            recover_pc_input[:, :, 2] = recover_pc[:, :, 1].clone()
            recover_pc_input = recover_pc_input.transpose(2, 1).contiguous()

            pred_def2, tran_def2, global_feat_def2 = net_c(recover_pc_input)
            optimizer_c.zero_grad()
            loss_c = cls_loss(pred_def1, label) + cls_loss(pred_def2, label)
            loss_c.backward()
            optimizer_c.step()
            train_loss_aug_cls.append(loss_c.detach().cpu().numpy())
            preds = pred_def1.max(dim=1)[1]
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        if optimizer_c.param_groups[0]['lr'] > 1e-5:
            scheduler_c.step()
        if optimizer_c.param_groups[0]['lr'] < 1e-5:
            for param_group in optimizer_c.param_groups:
                param_group['lr'] = 1e-5
        log_string("loss cls aug: %f" % np.mean(train_loss_aug_cls))
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        test_acc = eval_one_epoch(net_c.eval(), dataloader_cls_test)
        log_string("aug samples: train_acc: %f  test_acc: %f" % (train_acc, test_acc))
        if test_acc > best_acc:
            best_acc = max(best_acc, test_acc)
            if best_acc > 0.9:
                savepath = str(checkpoints_dir) + '/best_model_%f.pth' % best_acc
                state = {
                    'model_state_dict_c': net_c.state_dict(),
                    'optimizer_state_dict_c': optimizer_c.state_dict(),
                    'model_state_dict_a': net_a.state_dict(),
                    'optimizer_state_dict_a': optimizer_a.state_dict()
                }
                torch.save(state, savepath)
        log_string("best_acc: %f"  %
                   (best_acc))

    # training net_c with clean samples
    if opt.include_original:
        for clean_i in range(inner_iter):
            epoch += 1
            net_c.train()
            train_loss_clean = []
            train_pred = []
            train_true = []
            for i, data in tqdm(enumerate(dataloader_cls_train), total=len(dataloader_cls_train), smoothing=0.9):
                points, label = data
                label = label[:, 0]
                points, label = points.cuda().float(), label.cuda().long()
                points = PointcloudScaleAndTranslate(points)
                points = points.transpose(2, 1).contiguous()
                pred_ori, tran_ori, global_feat_ori = net_c(points)
                optimizer_c.zero_grad()
                loss_c = cls_loss(pred_ori, label)
                loss_c.backward()
                optimizer_c.step()
                train_loss_clean.append(loss_c.detach().cpu().numpy())
                preds = pred_ori.max(dim=1)[1]
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

            if optimizer_c.param_groups[0]['lr'] > 1e-5:
                scheduler_c.step()
            if optimizer_c.param_groups[0]['lr'] < 1e-5:
                for param_group in optimizer_c.param_groups:
                    param_group['lr'] = 1e-5
            log_string("loss cls clean: %f" % np.mean(train_loss_clean))
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_acc = metrics.accuracy_score(train_true, train_pred)
            test_acc = eval_one_epoch(net_c.eval(), dataloader_cls_test)
            log_string("clean samples: train_acc: %f  test_acc: %f" % (train_acc, test_acc))
            if test_acc > best_acc:
                best_acc = max(best_acc, test_acc)
                if best_acc > 0.9:
                    savepath = str(checkpoints_dir) + '/best_model_%f.pth' % best_acc
                    state = {
                        'model_state_dict_c': net_c.state_dict(),
                        'optimizer_state_dict_c': optimizer_c.state_dict(),
                        'model_state_dict_a': net_a.state_dict(),
                        'optimizer_state_dict_a': optimizer_a.state_dict()
                    }
                    torch.save(state, savepath)
            log_string("best_acc: %f"  %
                       (best_acc))

    # train net_a
    for aug_train_i in range(inner_iter):
        log_string('Outer iteration %d (%d/%s):' % (outer_iter + 1, outer_iter + 1, opt.outer))
        epoch += 1
        net_c.train()
        net_a.train()
        train_loss_aug = []

        for i, data in tqdm(enumerate(dataloader_train_aug), total=len(dataloader_train_aug), smoothing=0.9):
            src_pc, key_pts, w_pc, metahandles, coef_feat, label = data
            src_pc, key_pts, w_pc, metahandles, coef_feat, label = src_pc.cuda().float(), \
                                                                   key_pts.cuda().float(), \
                                                                   w_pc.cuda().float(), \
                                                                   metahandles.cuda().float(), \
                                                                   coef_feat.cuda().float(), \
                                                                   label.cuda().squeeze().to(torch.int64)
            metahandles = metahandles.view(-1, 15, 150)
            B, N, K = w_pc.shape

            if opt.offset:
                random_coef = torch.normal(0, opt.std, size=(B, opt.num_basis, 1, 1)).cuda()
            else:
                random_coef = torch.normal(0, opt.std, size=(B, opt.num_basis, 1, K)).cuda()
            def_key_pts, def_pc, coef = net_a(coef_feat, random_coef, metahandles, key_pts, w_pc)
            recover_pc = torch.bmm(w_pc, key_pts)
            cd_loss = chamfer_distance(recover_pc.clone(), def_pc.clone())
            cd_loss = cd_loss.mean()


            def_pc1 = def_pc.clone()
            def_pc_sym_y = def_pc1 * \
                           torch.tensor([-1, 1, 1]).cuda()
            def_pc_sym_x = def_pc1 * \
                           torch.tensor([1, -1, 1]).cuda()
            sym_loss = torch.minimum(chamfer_distance(def_pc1, def_pc_sym_x),
                                     chamfer_distance(def_pc1, def_pc_sym_y))
            sym_loss = sym_loss.mean()

            def_pc_input = def_pc.clone()
            def_pc_input[:, :, 1] = def_pc[:, :, 2].clone()
            def_pc_input[:, :, 2] = def_pc[:, :, 1].clone()
            def_pc_input = def_pc_input.transpose(2, 1).contiguous()

            recover_pc_input = recover_pc.clone()
            recover_pc_input[:, :, 1] = recover_pc[:, :, 2].clone()
            recover_pc_input[:, :, 2] = recover_pc[:, :, 1].clone()
            recover_pc_input = recover_pc_input.transpose(2, 1).contiguous()

            pred_def1, tran_def1, global_feat_def1 = net_c(def_pc_input)
            pred_def2, tran_def2, global_feat_def2 = net_c(recover_pc_input)

            optimizer_a.zero_grad()
            loss_aug = -cls_loss(pred_def1, label) * opt.adv_weight + cd_loss * opt.cd_weight + sym_loss * opt.sym_weight
            loss_aug.backward()
            optimizer_a.step()
            train_loss_aug.append(loss_aug.detach().cpu().numpy())
        log_string("loss aug: %f" % np.mean(train_loss_aug))
        scheduler_a.step()


    savepath = str(checkpoints_dir) + '/last_model.pth'
    state = {
        'model_state_dict_c': net_c.state_dict(),
        'optimizer_state_dict_c': optimizer_c.state_dict(),
        'model_state_dict_a': net_a.state_dict(),
        'optimizer_state_dict_a': optimizer_a.state_dict()
    }
    torch.save(state, savepath)



if opt.finetune:
    if best_acc > 0.9:
        checkpoint = torch.load(str(experiment_dir) +
                                '/checkpoints/best_model_%f.pth' % best_acc)
        net_c.load_state_dict(checkpoint['model_state_dict_c'])

    optimizer_c = torch.optim.Adam(
        net_c.parameters(),
        lr=opt.learning_rate * opt.finetune_lr_scale,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=opt.decay_rate)
    scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_c, step_size=20, gamma=opt.lr_decay)

    # training net_c with clean samples
    for clean_i in range(100):
        epoch += 1
        scheduler_c.step()
        net_c.train()
        train_loss_clean = []
        train_pred = []
        train_true = []
        for i, data in tqdm(enumerate(dataloader_cls_train), total=len(dataloader_cls_train), smoothing=0.9):
            points, label = data
            label = label[:, 0]
            points, label = points.cuda().float(), label.cuda().long()
            points = PointcloudScaleAndTranslate(points)
            points = points.transpose(2, 1).contiguous()
            pred_ori, tran_ori, global_feat_ori = net_c(points)
            optimizer_c.zero_grad()
            loss_c = cls_loss(pred_ori, label)
            loss_c.backward()
            optimizer_c.step()
            train_loss_clean.append(loss_c.detach().cpu().numpy())
            preds = pred_ori.max(dim=1)[1]
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        log_string("loss cls finetune: %f" % np.mean(train_loss_clean))
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        test_acc = eval_one_epoch(net_c.eval(), dataloader_cls_test)
        log_string("finetune: train_acc: %f  test_acc: %f" % (train_acc, test_acc))

        if optimizer_c.param_groups[0]['lr'] > 1e-5:
            scheduler_c.step()
        if optimizer_c.param_groups[0]['lr'] < 1e-5:
            for param_group in optimizer_c.param_groups:
                param_group['lr'] = 1e-5

        if test_acc > best_acc:
            best_acc = max(best_acc, test_acc)
            if best_acc > 0.9:
                savepath = str(checkpoints_dir) + '/finetune_best_model_%f.pth' % best_acc
                state = {
                    'model_state_dict_c': net_c.state_dict(),
                    'optimizer_state_dict_c': optimizer_c.state_dict(),
                    'model_state_dict_a': net_a.state_dict(),
                    'optimizer_state_dict_a': optimizer_a.state_dict()
                }
                torch.save(state, savepath)
        log_string("best_acc: %f"  %
                   (best_acc))

    savepath = str(checkpoints_dir) + '/finetune_last_model.pth'
    state = {
        'model_state_dict_c': net_c.state_dict(),
        'optimizer_state_dict_c': optimizer_c.state_dict(),
        'model_state_dict_a': net_a.state_dict(),
        'optimizer_state_dict_a': optimizer_a.state_dict()
    }
    torch.save(state, savepath)