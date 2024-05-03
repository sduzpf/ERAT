from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import random
import argparse
import numpy as np
from models import *
from torch.autograd import Variable
import utils.dataloader_cifar as dataloader
import torchvision.transforms as transforms
import scipy.io as sio
from utils.autoaugment import CIFAR10Policy, ImageNetPolicy
from utils.step import LinfStep, L2Step


parser = argparse.ArgumentParser(description='Effective and Robust Adversarial Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate 0.02')
parser.add_argument('--noise_mode', default='instance', help='lable noise type', choices=[pairflip, symmetric,instance])
parser.add_argument('--poison_type', default='L2C', help='data poisoning type', choices=[C, L2C, P1, P2, P3, P4, P5])
parser.add_argument('--noise_rate', type=float, help='label noise rate', default=0.6)
parser.add_argument('--eps_a', type=float, default=0.032, help='attack budget for data poisoning')
parser.add_argument('--eps_d', type=float, default=0.032, help='defense defense for data poisoning')
parser.add_argument('--model', type=str, help='cnn,resnet', default='resnet34')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int, help='10, or 100')
parser.add_argument('--data_path', default='./Datasets/cifar-10-python.tar/cifar-10-python/cifar-10-batches-py',
                    type=str, help='path to dataset')
parser.add_argument('--dataset', type=str, help='cifar10, or cifar100', default='cifar10')
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='./Dual/w8/Par7')
parser.add_argument('--eta', type=float, default=0.1, help='Balancing class number')
parser.add_argument('--constraint_a', default='Linf', choices=['Linf', 'L2'], help='data poisoning constraint for attack', type=str)
parser.add_argument('--constraint_d', default='L2', choices=['Linf', 'L2'], help='data poisoning constraint for defense', type=str)
parser.add_argument('--poison_data_path', default='./CIFAR10Poison', type=str)


args = parser.parse_args()

#settings for imargary data poisoning
args.step_size = args.eps_a / 5
args.num_steps = 7
args.random_restarts = 1

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}

def batch_adv_attack(args, net, x, x1, labels):
    orig_x = x.clone().detach()
    step = STEPS[args.constraint_d](orig_x, args.eps_d, args.step_size)

    def get_adv_examples(x):
        for _ in range(args.num_steps):
            x = x.clone().detach().requires_grad_(True)
            logits = net(x)
            logits1 = net(x1)
            loss =  - nn.CrossEntropyLoss()(logits1, labels) - torch.mean((torch.softmax(logits, dim=1) - torch.softmax(logits1.detach(), dim=1)) ** 2)
            grad = torch.autograd.grad(loss, [x])[0]
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)
                x = torch.clamp(x, 0, 1)
        return x.clone().detach()

    to_ret = None

    if args.random_restarts == 0:
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    elif args.random_restarts == 1:
        x = step.random_perturb(x)
        x = torch.clamp(x, 0, 1)
        adv = get_adv_examples(x)
        to_ret = adv.detach()
    else:
        for _ in range(args.random_restarts):
            x = step.random_perturb(x)
            x = torch.clamp(x, 0, 1)

            adv = get_adv_examples(x)
            if to_ret is None:
                to_ret = adv.detach()

            logits = net(adv)
            corr, = accuracy(logits, target, topk=(1,), exact=True)
            corr = corr.bool()
            misclass = ~corr
            to_ret[misclass] = adv[misclass]

    return to_ret.detach().requires_grad_(False)


transform_strong_10 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
    ]
)

def strong_data_augment(image):
    image = image.cpu()
    for i in range(len(image)):
        image[i] = transform_strong_10(image[i])
    return image

# Training 
def train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader):
    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    i = 0
    for batch_idx, (inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x) in enumerate(labeled_trainloader):
        i = i +1
        try:
            inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = next(unlabeled_train_iter)
            
        batch_size = inputs_x.size(0)
        labels_x = labels_x.cuda()
        
        inputs_x, inputs_x2, inputs_x3, inputs_x4, labels_x = inputs_x.cuda(), inputs_x2.cuda(), inputs_x3.cuda(),\
                                                                   inputs_x4.cuda(), labels_x.cuda()
        inputs_u, inputs_u2, inputs_u3, inputs_u4, labels_u = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda(), inputs_u4.cuda(), labels_u.cuda()
            

        # adversarial training ----------------------------------------------------------------
        net.eval()  #uniformly poisoning data
        leng = int(batch_size/2)
        args.constraint_d = 'Linf'
        args.eps_d = 0.032
        inp_adv_x1 = batch_adv_attack(args, net, inputs_x[:leng], inputs_x2[:leng], labels_x[:leng])
        inp_adv_u1 = batch_adv_attack(args, net, inputs_u[:leng], inputs_u2[:leng], labels_u[:leng]) 
        
        args.constraint_d = 'L2'
        args.eps_d = 0.5
        inp_adv_x2 = batch_adv_attack(args, net, inputs_x[leng:], inputs_x2[leng:], labels_x[:leng])
        inp_adv_u2 = batch_adv_attack(args, net, inputs_u[leng:], inputs_u2[leng:], labels_u[:leng])
        net.train()

        inp_adv_x = torch.cat([inp_adv_x1, inp_adv_x2], dim=0)
        inp_adv_u = torch.cat([inp_adv_u1, inp_adv_u2], dim=0)

        with torch.no_grad():            
            # label guessing of unlabeled samples
            logits_u = net(inputs_u)
            pu = torch.softmax(logits_u, dim=1)
            ptu = pu ** (1 / args.T)  # temparature sharpening
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

        with torch.no_grad():
            inputs_x5 = strong_data_augment(inp_adv_x)
            inputs_u5 = strong_data_augment(inp_adv_u)
        inputs_x5, inputs_u5  = inputs_x5.cuda(), inputs_u5.cuda()

        # semi-supervised learning               
        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
        all_inputs_adv =  torch.cat([inp_adv_x, inputs_x5, inp_adv_u, inputs_u5], dim=0)

        all_targets = torch.cat([labels_x, labels_x, targets_u, targets_u], dim=0)
        all_inputs = Variable(all_inputs.data, requires_grad=True)
        

        logit = net(all_inputs)     
        logit_adv = net(all_inputs_adv)
        
        Lx_s1, Lu_s1, lamb_s1 = criterion(logit[:batch_size * 2],  all_targets[:batch_size * 2], logit[batch_size * 2:], all_targets[batch_size * 2:],
                                       epoch + batch_idx / num_iter, warm_up)
        Lx_s2, Lu_s2, lamb_s2 = criterion(logit_adv[:batch_size * 2],  all_targets[:batch_size * 2], logit_adv[batch_size * 2:], all_targets[batch_size * 2:],
                                       epoch + batch_idx / num_iter, warm_up)
                                                                                                                    
        Lx_all =  Lx_s1 + Lx_s2 
        Lu_all = lamb_s1 * Lu_s1 + lamb_s2 * Lu_s2 
        loss_all =  Lx_all + Lu_all

        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, args.noise_rate, args.noise_mode, epoch, args.num_epochs, batch_idx + 1,
                            num_iter, Lx_all.item(), Lu_all.item()))
        sys.stdout.flush()

def warmup(epoch, net, optimizer, dataloader):
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, inputs1, labels, _, path) in enumerate(dataloader):
        inputs, inputs1, labels = inputs.cuda(), inputs1.cuda(), labels.cuda()
        
        # adversarial training
        batch_size = inputs.size(0)
        leng = int(batch_size/2)
  
        net.eval() #uniformly poisoning data
        args.constraint_d = 'Linf'
        args.eps_d = 0.032
        inp_adv1 = batch_adv_attack(args, net, inputs[:leng], inputs1[:leng], labels[:leng])
    
        args.constraint_d = 'L2'
        args.eps_d = 0.5
        inp_adv2 = batch_adv_attack(args, net, inputs[leng:], inputs1[leng:], labels[leng:])
        net.train()       

        inp_adv = torch.cat([inp_adv1, inp_adv2], dim=0)    

        optimizer.zero_grad()
        logits = net(inputs)
        logits_adv = net(inp_adv)
        loss = CEloss(logits, labels) + CEloss(logits_adv, labels)
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.noise_rate, args.noise_mode, epoch, args.num_epochs, batch_idx + 1,
                            num_iter, loss.item()))
        sys.stdout.flush()


def test(epoch, net, best_acc_=0, save=False):
    net.eval()
    correct = 0
    correct_n = 0
    total = 0
    total_n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = net(inputs)
            _, predicted = torch.max(logits, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

        for batch_idx, (inputs_n, targets) in enumerate(testp_loader):
            inputs_n, targets_n = inputs_n.cuda(), targets.cuda()
            logits_n = net(inputs_n)
            _, predicted_n = torch.max(logits_n, 1)

            total_n += targets_n.size(0)
            correct_n += predicted_n.eq(targets_n).cpu().sum().item()

    acc = 100. * correct / total
    acc_n = 100. * correct_n / total
    if save:
        if acc > best_acc_:
            state = {'net_state_dict': net.state_dict(),
                     'epoch': epoch,
                     'acc': acc,
                     'acc_n': acc_n,
                     }
            torch.save(state, os.path.join(save_dir, str(args.noise_rate) + '_' + str(args.eta) + '_' +   'best.pth.tar'))
            best_acc_ = acc
        if epoch == args.num_epochs - 1:
            state = {'net_state_dict': net.state_dict(),
                     'epoch': epoch,
                     'acc': acc,
                     'acc_n': acc_n,
                     }
            torch.save(state, os.path.join(save_dir, str(args.noise_rate) + '_' + str(args.eta) + '_' +   'last.pth.tar'))

    return acc, acc_n, best_acc_

def eval_train(epoch, net, all_loss, all_loss_pred, class_p):
    net.eval()
    noise_label = torch.zeros(num_sample)
    losses = torch.zeros(num_sample)
    Ratios = np.zeros(args.num_class)
    Ba_num = np.zeros(args.num_class)
    Rate = np.zeros(args.num_class)
    class_loss = np.zeros(args.num_class)
    class_ind = {}
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, _, index) in enumerate(eval_loader):
            targetx = targets
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)
            targets_s = torch.zeros(batch_size, args.num_class).scatter_(1, targetx.view(-1, 1), 1)
            targets_s = targets_s.cuda()
            inputs = inputs.cuda()            
            logits = net(inputs)  
            loss = torch.sum((torch.softmax(logits, dim=1) - targets_s) ** 2, dim=1)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                noise_label[index[b]] = targets[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)
    if args.noise_rate == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)
    threshold = torch.mean(losses) #threshold for initial dataset separation
    prob_clean = (losses.data.numpy()<= threshold.item())
    clean_len =  np.sum(prob_clean).item()
    for k in range(args.num_class):
        class_ind[k] = [i for i, x in enumerate(noise_label) if x == k]
        class_loss[k] = torch.mean(losses[class_ind[k]])
        if epoch > warm_up:
            class_loss[k] = torch.mean(losses[class_p[k]])
        class_loss[k] = (1 - class_loss[k])**args.eta #Class_level divergence
    loss_avg =  np.sum(class_loss)
    for k in range(args.num_class):

        Ratios[k] = class_loss[k]/loss_avg # clean label assignment for each class
        Ba_num[k] = Ratios[k] *clean_len 
        prob_ = np.argsort(losses[class_ind[k]].cpu().numpy())
        class_p[k] = np.array(class_ind[k])[prob_[0:int(Ba_num[k])].astype(int)].squeeze()

    return class_p, Ba_num, losses, all_loss

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = CEloss(outputs_x, targets_x) 
        Lu = torch.mean((probs_u - targets_u) ** 2)
        return Lx, Lu, linear_rampup(epoch, warm_up)


save_dir = args.result_dir + '/' + args.dataset + '/' + args.model + '/' + args.noise_mode + '/' + args.poison_type + '/' + str(args.eps_a)
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
    
stats_log = open(save_dir + '/' + str(args.noise_rate) + '_' + str(args.eta)  + '_stats.txt', 'w')

txtfile = save_dir + '/' + str(args.noise_rate) + '_' + str(args.eta)  + '.txt'
if os.path.exists(txtfile):
    os.system('rm %s' % txtfile)
with open(txtfile, "a") as myfile:
    myfile.write('epoch: test_acc test_acc_n\n')


if args.dataset == 'cifar10':
    warm_up = 10
elif args.dataset == 'cifar100':
    warm_up = 30


loader = dataloader.cifar_dataloader(args.dataset, noise_mode=args.noise_mode,
                                     noise_rate=args.noise_rate, eps = args.eps_a,noise_type = args.poison_type, 
                                     batch_size=args.batch_size, num_workers=3, \
                                     root_dir=args.data_path, result_dir=args.result_dir, log=stats_log,
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.noise_rate, args.noise_mode))

poison_loader = dataloader.cifar_poisoneddataloader(args.dataset, noise_mode=args.noise_mode,
                                     noise_rate=args.noise_rate, eps = args.eps_a, batch_size=args.batch_size, num_workers=3, \
                                     root_dir=args.poison_data_path + '/' + str(args.eps_a), test_dir=args.poison_data_path + '/' + str(args.eps_a)+ '_test', 
                                     result_dir=args.result_dir, constraint = args.constraint_a,
                                     poison_type = args.poison_type, log=stats_log,
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.noise_rate, args.noise_mode))


print('| Building net')
if args.model == 'cnn':
    net = CNN(input_channel=3, n_outputs=args.num_classes).cuda()
else:
    net = ResNet34().cuda()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


CEloss = nn.CrossEntropyLoss()

all_loss = [[], []]  # save the history of losses from two networks
best_acc_ = 0.0
eval_acc = 0.0
eval_acc_1 = 0.0
eval_acc_2 = 0.0
class_p = {} #seperated clean labal index from the previous epoch
num_sample = 50000 #number of training data

test_loader = loader.run(0, 'test')
eval_loader = poison_loader.run(0, 'eval_train')
testp_loader = poison_loader.run(0, 'test')
    
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 60:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    if epoch < warm_up:
        warmup_trainloader = poison_loader.run(0, 'warmup')
        print('Warmup Net1')
        warmup(epoch, net, optimizer, warmup_trainloader)
        Pedictor_n.load_state_dict(Pedictor.state_dict())

    else:
        print('Data separation')
        class_p, Ba_num, prob, all_loss[0] = eval_train(epoch, net, all_loss[0], class_p)
        labeled_trainloader, unlabeled_trainloader = poison_loader.run(0, 'train', Ba_num, prob1)
        
        print('Train Net1')        
        train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader)

    test_acc, test_acc_n, best_acc_ = test(epoch, net, best_acc_=best_acc_, save=True, )   

    
    print('test acc on test images is ', test_acc)
    print('test acc on poisoned test images is ', test_acc_n)
    print('best acc on test images is ', best_acc_)
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': ' + str(test_acc) + ' ' + str(test_acc_n) + ' ' + str(best_acc_) + "\n")


