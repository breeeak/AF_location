import random
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
from msmodel import MsModel

import os
import time
import json
import shutil
import numpy as np
import cv2
import sklearn.metrics as skm
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy import signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from myutils import Logger, AverageMeter, accuracy, mkdir_p, savefig


def generate_stft_fig(data, fs=100, features=2, image_size=224, win_sz=100, overlap=50):
    img_size = (image_size, image_size)
    # dpi fix
    fig = plt.figure(frameon=False)
    dpi = fig.dpi
    # fig size / image size
    images = []
    for i in range(len(data)):
        for m in range(features):
            win = signal.windows.hann(win_sz)
            f, t, zxx = signal.stft(data[i][:, m], fs, window=win, nperseg=win_sz, noverlap=overlap, nfft=win_sz,
                                    return_onesided=True, boundary='zeros', padded=True, axis=- 1)

            # plt.figure(figsize=(33,21))
            fig = Figure()
            fig.subplots_adjust(0, 0, 1, 1)
            #         fig.add_axes([0,0,1,1])
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            ax.pcolormesh(t, f, 20 * np.log10(np.abs(zxx)), shading='gouraud')
            ax.axis('off')
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            # #         if name%10 == 0:
            c = cv2.resize(image, img_size)
            images.append(c)
    return images


class Ecg_loader(torch.utils.data.Dataset):
    def __init__(self, data_root, features):
        super(Ecg_loader, self).__init__()
        self.idx2name = l2n = {'N': 0, 'A': 1}
        self.num_classes = len(self.idx2name)
        self.features = features
        self.inputs = []
        self.labels = []
        data_list = os.listdir(data_root)
        # i = 0
        for record_name in tqdm(data_list):
            file_path = os.path.join(data_root, record_name)
            l = np.load(os.path.join(file_path, record_name + '_label.npy'))
            sigs_path = []
            for i in range(len(l)):
                sig_path = os.path.join(file_path, str(record_name) + "_sig_" + str(i) + '.npy')
                sigs_path.append(sig_path)
            self.inputs.extend(sigs_path)
            self.labels.extend(l)
        pass


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sig = np.load(self.inputs[idx])
        sigs = []
        for i in range(self.features):
            sigs.append(sig[:, i])
        ss = np.expand_dims(np.array(sigs).flatten(), axis=0)
        x = torch.from_numpy(ss).float()
        label = np.array(self.labels[idx], dtype=np.int64)
        # y = torch.nn.functional.one_hot(torch.from_numpy(label), self.num_classes).long()
        y = torch.from_numpy(label).long()
        return x, y


def evaluate(outputs, labels, label_names=None):
    gt = torch.cat(labels, dim=0)
    pred = torch.cat(outputs, dim=0)
    probs = pred
    pred = torch.argmax(pred, dim=1)
    acc = torch.div(100 * torch.sum((gt == pred).float()), gt.shape[0])
    name_dict = {0: 'Normal beat (N)', 1: 'Atrial fibrillation beat (A)'}
    print('accuracy :', acc)

    gt = gt.cpu().tolist()
    pred = pred.cpu().tolist()

    report = skm.classification_report(
        gt, pred,
        target_names=[name_dict[i] for i in np.unique(gt)],
        digits=3)
    scores = skm.precision_recall_fscore_support(
        gt,
        pred,
        average=None)
    print(report)
    print("F1 Average {:3f}".format(np.mean(scores[2][:3])))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = np.unique(gt).shape[0]
    oh_gt = np.zeros((len(gt), n_classes))
    plt.figure()
    colors = ['b', 'g', 'r', 'c']

    for i in range(n_classes):
        oh_gt[:, gt == i] = 1

        fpr[i], tpr[i], _ = roc_curve(gt, probs[:, i].cpu(), pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        lw = 2
        plt.plot(fpr[i], tpr[i], color=colors[i],
                 lw=lw, label=name_dict[i] + ' : %0.4f' % roc_auc[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class-Wise AUC and ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.checkpoint, 'roc.png'))
    return 0


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        prec1 = accuracy(outputs.data, targets.data)
        if float(torch.__version__[:3]) < 0.5:
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))
        else:
            losses.update(loss.data, inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # evaluate(pred, gt)
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, label_names=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    gt = []
    pred = []
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        gt.append(targets.data)
        pred.append(outputs.data)
        prec1 = accuracy(outputs.data, targets.data)
        if float(torch.__version__[:3]) < 0.5:
            losses.update(loss.data[0], inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))
        else:
            losses.update(loss.data, inputs[0].size(0))
            top1.update(prec1[0], inputs[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress

    evaluate(pred, gt, label_names=label_names)
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global lr
    if epoch in args.schedule:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class Args():
    manualSeed = 1

    checkpoint = "checkpoint"  # checkpoint directory
    resume = ""  # path to latest checkpoint (default: none)

    data = "E:\\1_dataset\\CPSC\\data_prepare_seg_test"  # path of dataset
    features = 2
    evaluate = False  # evaluate model on validation set
    transformation = None
    workers = 8

    depth = 110  # model depth
    block_name = 'BasicBlock'  # the building block for Resnet and Preresnet: BasicBlock, Bottleneck

    epochs = 1
    start_epoch = 0
    train_batch = 64
    test_batch = 64
    lr = 0.001
    weight_decay = 5e-4
    gamma = 0.1  # 'LR is multiplied by gamma on schedule.'
    schedule = [150, 225]  # Decrease learning rate at these epochs


args = Args()
lr = args.lr

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    # Data
    print('==> Preparing dataset ')

    dataloader = Ecg_loader
    train_path = args.data

    traindir = os.path.join(train_path, 'train')
    valdir = os.path.join(train_path, 'val')
    if not args.evaluate:
        trainset = dataloader(traindir, features=args.features)
    testset = dataloader(valdir, features=args.features)

    idx2name = testset.idx2name
    label_names = idx2name.keys()
    num_classes = len(label_names)

    if not args.evaluate:
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model MsModel")
    model = MsModel(num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Resume
    title = 'ecg-resnet' + str(args.depth)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda, label_names=label_names)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, label_names=label_names)

        # append logger file
        logger.append([lr, train_loss.cpu(), test_loss.cpu(), train_acc.cpu(), test_acc.cpu()])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        # torch.save(model_object, 'model.pkl')
        # model = torch.load('model.pkl')

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


if __name__ == '__main__':
    main()
