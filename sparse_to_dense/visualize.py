import utils
import criteria
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
from metrics import AverageMeter, Result
from models import ResNet
import os
import time
import csv
import pathlib

import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import dataset_3d.utils.visualization_utils as vis_utils

cudnn.benchmark = True


args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
              'delta1', 'delta2', 'delta3',
              'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()


def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    traindir = os.path.join('data', args.data, 'train')
    valdir = os.path.join('data', args.data, 'val')

    print("train =", traindir)
    print("val =", valdir)
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(
            num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(
            num_samples=args.num_samples, max_depth=max_depth)

    if args.data == 'nyudepthv2':
        from dataloaders.nyu_dataloader import NYUDataset
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                                       modality=args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
                                 modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                                         modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
                                   modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'carla':
        # TODO pass sensor names as arguments
        from dataloaders.carla_dataloader import CarlaDataset
        if not args.evaluate:
            train_dataset = CarlaDataset(traindir, type='train',
                                         modality=args.modality, max_depth=max_depth)
        val_dataset = CarlaDataset(valdir, type='val',
                                   modality=args.modality, max_depth=max_depth)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    torch.manual_seed(0)
    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id: np.random.seed(work_id))
        # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader


def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
    print("=> loading best model '{}'".format(args.evaluate))
    checkpoint = torch.load(args.evaluate)
    output_directory = os.path.dirname(args.evaluate)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch'] + 1
    best_result = checkpoint['best_result']
    model = checkpoint['model']
    print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    _, val_loader = create_data_loaders(args)
    args.evaluate = True
    visualize(val_loader, model, output_directory)


def visualize(val_loader, model, output_directory, max_depth=120, vis_amount=50):
    model.eval()  # switch to evaluate mode

    # create a visualization folder
    out_base = pathlib.Path(output_directory)
    out_vis_folder = out_base.joinpath("visualization")
    out_vis_folder.mkdir(exist_ok=True)

    # pick n random indices
    rng = np.random.default_rng()
    sample_idxs = rng.integers(len(val_loader), size=vis_amount)

    def save_depth(depth_map, name, idx):
        path = str(out_vis_folder.joinpath(
            "{}_depth_{}.png".format(idx, name)))
        cv2.imwrite(path, depth_map)

    # for i, (input, target) in enumerate(val_loader):
    for i, (input, target) in enumerate(val_loader):

        if i > vis_amount:
            break

        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()

        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()

        rgb = input[:, :3, :, :]
        depth_input = input[:, 3:, :, :]
        depth_gt = target
        depth_pred = pred

        rgb = 255 * np.transpose(np.squeeze(rgb.cpu().numpy()), (1, 2, 0))
        depth_input = np.squeeze(depth_input.cpu().numpy())
        depth_gt = np.squeeze(depth_gt.cpu().numpy())
        depth_pred = np.squeeze(depth_pred.data.cpu().numpy())

        depth_gt_vis = vis_utils.depth_to_img(
            depth_gt, max_depth=max_depth)

        depth_input_vis = vis_utils.depth_to_img(
            depth_input, max_depth=max_depth)

        depth_pred_vis = vis_utils.depth_to_img(
            depth_pred, max_depth=max_depth)

        # save the rgb image
        rgb_path = str(out_vis_folder.joinpath("{}_rgb.png".format(i)))
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        save_depth(depth_gt_vis, "gt", i)
        save_depth(depth_input_vis, "lidar", i)
        save_depth(depth_pred_vis, "s2d", i)


if __name__ == '__main__':
    main()
