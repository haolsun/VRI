from meta_net import vri, vri_dec, vri_prior
from wideresnet import WideResNet
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataloader import CIFAR10, CIFAR100
import argparse
import os, warnings
from rand_aug import RandAugmentMC
import numpy as np
from utilities import *
from tqdm import tqdm
import datetime

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.4, help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=170, type=int, help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int, help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=100, type=int, help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int, help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str, help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--var', type=float, default=0.1, help='Pre-fetching threads.')
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--lam', type=float, default=0.001)
parser.add_argument('--alpha', default=0.2, type=float, help='parameter for Beta in mixup')
parser.set_defaults(augment=True)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid


def build_dataset(root, args):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize((0.507,0.487,0.441),(0.267,0.265,0.276))
    if args.augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),  (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    strong_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor()
    ])

    if args.dataset == 'cifar10':

        train_data_meta = CIFAR10(
                root=root, train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
                corruption_type=args.corruption_type, transform=train_transform, download=True, strong_t=None, normalize=normalize)

        train_data = CIFAR10(
                root=root, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
                corruption_type=args.corruption_type, transform=strong_transform, download=True, seed=args.seed,
            strong_t=strong_transform, normalize=normalize)

        test_data = CIFAR10(root=root, train=False, transform=test_transform, download=True, normalize=normalize)


    elif args.dataset == 'cifar100':
        train_data_meta = CIFAR100(
            root=root, train=True, meta=True, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=train_transform, download=True, strong_t=None, normalize=normalize)
        train_data = CIFAR100(
            root=root, train=True, meta=False, num_meta=args.num_meta, corruption_prob=args.corruption_prob,
            corruption_type=args.corruption_type, transform=strong_transform, download=True, seed=args.seed,
            strong_t=strong_transform, normalize=normalize)
        test_data = CIFAR100(root=root, train=False, transform=test_transform, download=True, normalize=normalize)

    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
            train_data_meta, batch_size=args.batch_size, shuffle=True,
            num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader

def build_model(args):
    model = WideResNet(depth=28, num_classes=args.dataset == 'cifar10' and 10 or 100, widen_factor=10)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def mixup(net, data_loader, optimizer, alpha, num_classes):

    net.train()
    num_iter = (len(data_loader.dataset) // data_loader.batch_size) + 1
    losses = 0.0

    for batch_idx, (inputs, labels, path) in enumerate(data_loader):#(tqdm.tqdm(data_loader, ncols=0)):
        l = np.random.beta(alpha, alpha)
        labels = torch.nn.functional.one_hot(labels.long(), num_classes).float()
        inputs, labels = inputs.cuda(), labels.cuda()

        idx = torch.randperm(inputs.size(0))

        input_a, input_b = inputs, inputs[idx]
        target_a, target_b = labels, labels[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits, _ = net(mixed_input)
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss

    return losses/num_iter

def warm_up(model, warm_loader, optimizer):
    model.train()
    acc_train = 0.0
    train_loss = 0.0
    for batch_idx, (inputs, targets, index) in enumerate(tqdm(warm_loader, ncols=0)):
        num = batch_idx
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, _ = model(inputs)
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]
        acc_train += prec_train
        loss = F.cross_entropy(outputs, targets.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        return train_loss/(num+1), acc_train/(num+1)


def train(train_loader, train_meta_loader, model, vnet, vnet_prior, optimizer_model, optimizer_vnet, optimizer_vnet_prior, epoch):
    print('Epoch: %d, lr: %.5f' % (epoch, optimizer_model.param_groups[0]['lr']))

    train_loss = 0
    meta_loss = 0
    acc_meta = 0.0
    acc_train = 0.0
    var_norms = 0

    num = 0
    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, path) in enumerate(tqdm(train_loader, ncols=0)):
        num = batch_idx
        model.train()

        meta_model = build_model(args).cuda()
        meta_model.load_state_dict(model.state_dict())

        oringal_targets = targets.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).float().cuda()

        # ========================== step 1 ====================================
        outputs, feat = meta_model(inputs)
        mean, log_var, v_lambda, _ = vnet(feat.detach(), targets)
        mean_p, log_var_p, _ = vnet_prior(feat.detach())
        l_f_meta = loss_function(v_lambda * outputs, targets_onehot) + + args.lam * kl_loss(mean, log_var, mean_p, log_var_p)

        # updata copy_model`s params
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = optimizer_model.param_groups[0]['lr']
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        # ========================= step 2 =====================================
        try:
            inputs_val, targets_val, _ = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val, _ = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()  # [500,3,32,32], [500]

        y_g_hat, _ = meta_model(inputs_val)
        prec_train = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]
        acc_meta += prec_train

        l_g_meta = F.cross_entropy(y_g_hat, targets_val.long())

        # update vnet params
        optimizer_vnet.zero_grad()
        optimizer_vnet_prior.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()
        optimizer_vnet_prior.step()

        # ========================= step 3 =====================================
        outputs, feat = model(inputs)
        prec_train = accuracy(outputs.data, oringal_targets.data, topk=(1,))[0]
        acc_train += prec_train

        with torch.no_grad():
            _, _, w_new, var_norm = vnet(feat.detach(), targets)

        loss = loss_function(w_new*outputs, targets_onehot)

        # update model params
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()
        var_norms += var_norm.item()

    return train_loss/(num+1), meta_loss/(num+1), acc_train/(num+1), acc_meta/(num+1), var_norms/(num+1)

def build_training(args):
    model = build_model(args).cuda()
    vnet = vri(704, 1024, 512, num_classes).cuda()
    vnet_prior = vri_prior(640, 1024, num_classes).cuda()

    optimizer_model = torch.optim.SGD(model.params(), 0.02, momentum=args.momentum, weight_decay=args.weight_decay)
    sch_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=5, eta_min=1e-4)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 3e-4, weight_decay=args.weight_decay)
    optimizer_vnet_prior = torch.optim.Adam(vnet_prior.params(), 3e-4, weight_decay=args.weight_decay)

    return model, vnet, vnet_prior, \
           optimizer_model, \
           sch_lr, optimizer_vnet, optimizer_vnet_prior

def test_ensembel(model_1, model_2, test_loader):
    model_1.eval()
    model_2.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_1, _ = model_1(inputs)
            outputs_2, _ = model_2(inputs)
            outputs = (outputs_1 + outputs_2) / 2
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy

def main():
    torch.manual_seed(args.seed)
    root = '../../data'
    train_loader_1, train_meta_loader_1, _ = build_dataset(root, args)
    train_loader_2, train_meta_loader_2, test_loader = build_dataset(root, args)

    global  num_classes
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100

    model_1, vnet_1, vnet_prior_1, optimizer_model_1, sch_lr_1, optimizer_vnet_1, optimizer_vnet_prior_1 \
        = build_training(args)
    model_2, vnet_2, vnet_prior_2, optimizer_model_2, sch_lr_2, optimizer_vnet_2, optimizer_vnet_prior_2 \
        = build_training(args)

    best_acc, meta_loss, acc_meta, var_norms = 0.0, 0.0, 0.0, 0.0

    for epoch in range(args.epochs):
        train_loss, meta_loss, acc_train, acc_meta, var_norms = train(train_loader_1, train_meta_loader_1, model_1,
                                                                      vnet_1,
                                                                      vnet_prior_1, optimizer_model_1, optimizer_vnet_1,
                                                                      optimizer_vnet_prior_1, epoch)
        train_loss, meta_loss, acc_train, acc_meta, var_norms = train(train_loader_2, train_meta_loader_2, model_2,
                                                                      vnet_2,
                                                                      vnet_prior_2, optimizer_model_2, optimizer_vnet_2,
                                                                      optimizer_vnet_prior_2, epoch)

        test_acc = test_ensembel(model_1, model_2, test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
        sch_lr_1.step()
        sch_lr_2.step()

        print(
            "epoch:[%d/%d]\t train_loss:%.4f\t meta_loss:%.4f\t train_acc:%.4f\t meta_acc:%.4f\t test_acc:%.4f\t var_norm:%.4f\t" % (
                (epoch + 1), args.epochs, train_loss, meta_loss, acc_train, acc_meta, test_acc, var_norms))
        print(
            "epoch:[%d/%d]\t, train_loss:%.4f\t, meta_loss:%.4f\t, train_acc:%.4f\t, meta_acc:%.4f\t, test_acc:%.4f\t var_norm:%.4f\t" % (
            (epoch + 1), args.epochs, train_loss, meta_loss, acc_train, acc_meta, test_acc, var_norms), file=mytxt)

        # mixup enhance
        if epoch % 5 == 0:
            train_loss = mixup(model_1, train_meta_loader_1, optimizer_model_1, args.alpha, num_classes)
            train_loss = mixup(model_2, train_meta_loader_2, optimizer_model_2, args.alpha, num_classes)
            test_acc = test_ensembel(model_1, model_2, test_loader)
            if test_acc >= best_acc:
                best_acc = test_acc
            print(
                "mixup epoch:[%d/%d]\t train_loss:%.4f\t test_acc:%.4f\t" % (
                    (epoch + 1), args.epochs, train_loss, test_acc))
            print(
                "mixup epoch:[%d/%d]\t, train_loss:%.4f\t, test_acc:%.4f\t" % (
                    (epoch + 1), args.epochs, train_loss, test_acc), file=mytxt)

    print('best_acc: ', best_acc)
    print('best_acc: ', best_acc, file=mytxt)


if __name__ == '__main__':
    save_path_dir = './exp_results/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)

    txt_name = args.dataset + '_' + args.corruption_type + '_' + str(args.corruption_prob) \
               + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(txt_name)
    mytxt = open(save_path_dir + txt_name + '.txt', mode='a', encoding='utf-8')
    print(args, file=mytxt)

    main()
    mytxt.close()