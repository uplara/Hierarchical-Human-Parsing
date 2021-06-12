import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data

# from dataset.data_pascal import DataGenerator
from dataset.dataset_tfrecord import define_dataset
from network.baseline import get_model
from progress.bar import Bar
from utils.gnn_loss import gnn_loss_noatt as ABRLovaszLoss

from utils.metric import *
from utils.parallel import DataParallelModel, DataParallelCriterion
from utils.visualize import inv_preprocess, decode_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation')
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    # Datasets
    parser.add_argument('--root', default='./data/Person', type=str)
    parser.add_argument('--val-root', default='./data/Person', type=str)
    parser.add_argument('--lst', default='./dataset/Pascal/train_id.txt', type=str)
    parser.add_argument('--val-lst', default='./dataset/Pascal/val_id.txt', type=str)
    parser.add_argument('--crop-size', type=int, default=473)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--hbody-cls', type=int, default=3)
    parser.add_argument('--fbody-cls', type=int, default=2)
    # Optimization options
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--lr-mode', type=str, default='poly')
    parser.add_argument('--ignore-label', type=int, default=255)
    # Checkpoints
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--restore_from', default='gs://vinit_helper/grapy/hierarchical_human_parsing/exp_v2/baseline64_miou.pth', type=str, help='provide google cloud link')
    parser.add_argument('--snapshot_dir', type=str, default='./checkpoints/exp/')
    parser.add_argument('--exp_path', type=str, default='gs://vinit_helper/grapy/hierarchical_human_parsing/exp_v2')
    parser.add_argument('--log-dir', type=str, default='./runs/')
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--save-num', type=int, default=2)
    # Misc
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method='poly'):
    if method == 'poly':
        current_step = epoch * iters_per_epoch + i_iter
        max_step = args.epochs * iters_per_epoch
        lr = args.learning_rate * ((1 - current_step / max_step) ** 0.9)
    else:
        lr = args.learning_rate
    optimizer.param_groups[0]['lr'] = lr
    return lr


def main(args):
    # initialization
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.method))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    # conduct seg network
    model = get_model(num_classes=args.num_classes)

    if args.eval or args.resume:
        restore_model_link = args.restore_from.split("/")[-1]
        restore_model_link = os.path.join('/content/', restore_model_link)

        cmd = f"gsutil -m cp -r {args.restore_from} /content/"
        if not os.path.exists(restore_model_link):
            os.system(cmd)

        model.load_state_dict(torch.load(restore_model_link, map_location='cpu'))

    model.float()
    model.cuda()

    # define criterion & optimizer
    criterion = ABRLovaszLoss(adj_matrix = torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]]), ignore_index=args.ignore_label, only_present=True, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], cls_p= args.num_classes, cls_h= args.hbody_cls, cls_f= args.fbody_cls)

    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    # key points
    best_val_mIoU = 0
    best_val_pixAcc = 0
    start = time.time()

    tfrecord_path = "gs://labelling-tools-data/tfrecords/person-tfrecord-v1.6_all_images.record"
    trainset, trainset_length = define_dataset(tfrecord_path, args.batch_size, train=True)
    valset, valset_length = define_dataset(tfrecord_path, args.batch_size, train=False)

    print("Training Volume", trainset_length)
    print("Validation Volume", valset_length)
    exp_path = args.exp_path

    if args.eval == True:
        valpath = os.path.join(args.snapshot_dir, 'test')
        if not os.path.exists(valpath):
            os.makedirs(valpath)
        validation(model, valset, 0, args.batch_size, valset_length, valpath, exp_path)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            print('\n{} | {}'.format(epoch, args.epochs - 1))
            # training
            _ = train(model, trainset, epoch, criterion, optimizer, writer, args.batch_size, trainset_length)

            if epoch%5 == 0:
                valpath = os.path.join(args.snapshot_dir, 'val_'+str(epoch))
                if not os.path.exists(valpath):
                    os.makedirs(valpath)
                validation(model, valset, epoch, args.batch_size, valset_length, valpath, exp_path)

            if epoch%2 == 0:
                model_dir = os.path.join(args.snapshot_dir, args.method + str(epoch) + '_miou.pth')
                torch.save(model.state_dict(), model_dir)
                print('Model saved to %s' % model_dir)

                cmd = f"gsutil -m cp -r {model_dir} {exp_path}"
                os.system(cmd)


def train(model, trainset, epoch, criterion, optimizer, writer, batch_size, trainset_length):
    # set training mode
    model.train()
    train_loss = 0.0
    iter_num = 0

    from tqdm import tqdm

    train_iterator = iter(trainset)
    num_iterations = int(trainset_length/batch_size)

    tbar = tqdm(range(num_iterations))
    for i_iter in tbar:
        sys.stdout.flush()
        start_time = time.time()
        iter_num += 1
        # adjust learning rate
        iters_per_epoch = trainset_length
        lr = adjust_learning_rate(optimizer, epoch, i_iter, iters_per_epoch, method=args.lr_mode)

        image, data = next(train_iterator)
        labels = data['segmentations']
        hlabel = data['segmentations_half']
        flabel = data['segmentations_full']

        image = torch.tensor(image.numpy()).permute(0, 3, 1, 2)
        labels = torch.tensor(labels.numpy()).permute(0, 3, 1, 2)
        hlabel = torch.tensor(hlabel.numpy()).permute(0, 3, 1, 2)
        flabel = torch.tensor(flabel.numpy()).permute(0, 3, 1, 2)

        image, labels, hlabel, flabel = image.cuda(), labels.long().cuda(), hlabel.cuda(), flabel.cuda()
        torch.set_grad_enabled(True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output loss
        # import pdb; pdb.set_trace()
        preds = model(image)
        loss = criterion(preds, [labels, hlabel, flabel])  # batch mean
        train_loss += loss.item()

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        if i_iter % 10 == 0:
            writer.add_scalar('learning_rate', lr, iter_num + epoch * trainset_length)
            writer.add_scalar('train_loss', train_loss / iter_num, iter_num + epoch * trainset_length)

        batch_time = time.time() - start_time
        # plot progress
        tbar.set_description('{} / {} | Time: {batch_time:.4f} | Loss: {loss:.4f}'.format(iter_num, trainset_length,
                                                                                  batch_time=batch_time,
                                                                                  loss=train_loss / iter_num))
        # bar.suffix = '{} / {} | Time: {batch_time:.4f} | Loss: {loss:.4f}'.format(iter_num, len(train_loader),
        #                                                                           batch_time=batch_time,
        #                                                                           loss=train_loss / iter_num)
        # bar.next()

    epoch_loss = train_loss / iter_num
    writer.add_scalar('train_epoch_loss', epoch_loss, epoch)
    tbar.close()
    # bar.finish()

    return epoch_loss


def validation(model, valset, epoch, batch_size, valset_length, valpath, exp_path):

    # set evaluate mode
    model.eval()

    # Iterate over data.
    from tqdm import tqdm

    val_iterator = iter(valset)
    num_iterations = int(valset_length/batch_size)
    tbar = tqdm(range(num_iterations))

    for idx in tbar:
        # image, target, hlabel, flabel, _ = batch

        image, data = next(val_iterator)
        target = data['segmentations']
        hlabel = data['segmentations_half']
        flabel = data['segmentations_full']

        image = torch.tensor(image.numpy()).permute(0, 3, 1, 2)
        target = torch.tensor(target.numpy()).permute(0, 3, 1, 2)
        hlabel = torch.tensor(hlabel.numpy()).permute(0, 3, 1, 2)
        flabel = torch.tensor(flabel.numpy()).permute(0, 3, 1, 2)

        image, target, hlabel, flabel = image.cuda(), target.cuda(), hlabel.cuda(), flabel.cuda()
        with torch.no_grad():
            h, w = target.size(2), target.size(3)
            outputs = model(image)
            # outputs = gather(outputs, 0, dim=0)
            preds = F.interpolate(input=outputs[0][-1], size=(h, w), mode='bilinear', align_corners=True)
            preds_hb = F.interpolate(input=outputs[1][-1], size=(h, w), mode='bilinear', align_corners=True)
            preds_fb = F.interpolate(input=outputs[2][-1], size=(h, w), mode='bilinear', align_corners=True)

            preds = torch.argmax(preds, 1)[0]
            preds_hb = torch.argmax(preds_hb, 1)[0]
            preds_fb = torch.argmax(preds_fb, 1)[0]

            fig = plt.figure(figsize=(12,10))
            plt.subplot(2,3,1)
            plt.imshow(image.permute(0, 2, 3, 1)[0].detach().cpu().numpy()*0.5+0.5)
            plt.title('Input Image', fontsize=20)
            plt.subplot(2,3,2)
            plt.imshow(preds.detach().cpu().numpy())
            plt.title('Predicted Grapy', fontsize=20)
            plt.subplot(2,3,3)
            plt.imshow(target.detach().cpu().numpy()[0,0,:,:])
            plt.title('Ground Truth Grapy', fontsize=20)
            plt.subplot(2,3,5)
            plt.imshow(preds_fb.detach().cpu().numpy())
            plt.title('Predicted Grapy Silhouette', fontsize=20)
            plt.subplot(2,3,6)
            plt.imshow(preds_hb.detach().cpu().numpy())
            plt.title('Predicted Half Grapy', fontsize=20)

            plt.savefig(os.path.join(valpath, str(idx)+'.png'))
            plt.close(fig)
    
    cmd = f"gsutil -m cp -r {valpath} {exp_path}"
    os.system(cmd)


if __name__ == '__main__':
    args = parse_args()
    main(args)