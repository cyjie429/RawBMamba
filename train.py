
from __future__ import print_function
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from data_feeder import ASVDataSet, load_data
from torch.utils.data import DataLoader
import numpy as np

import feature_extract
import math
from torch.autograd import Variable

from tqdm import tqdm


#from seBlock import SEBottle2neck
from mamba_ssm.models.ac_mamba import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
ids = [0]

# protocol_path
train_protocol = "/data2/chenyujie/data/ASVspoof2019LA/label/ASVspoof2019.LA.cm.train.trl.txt"
dev_protocol = "/data2/chenyujie/data/ASVspoof2019LA/label/ASVspoof2019.LA.cm.dev.trl.txt"
eval_protocol = "/data2/chenyujie/data/ASVspoof2021LA/CM/trial_metadata.txt"


class A_softmax(nn.Module):
    def __init__(self, gamma=0):
        super(A_softmax, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte().type(torch.bool)
        index = Variable(index)

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()

        return loss


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion=A_softmax()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr=optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion=A_softmax()
            loss = criterion(output, target)
            test_loss+=loss.item()
            result=output[0]
            pred = result.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def feval(model, feature_type, device):
    model.eval()
    name="eval.txt"
    result=[]
    scp=[]
    with torch.no_grad():
        wav_list, folder=load_data("eval", eval_protocol, mode="eval", feature_type="fft")
        '''
        frames=len(data)
        data=data.to(device)
        for idx in range(frames):
            output=model(data[idx])
            output=np.mean(output,axis=0)
            result.append(output)
        '''
        for idx in tqdm(range(len(wav_list)), desc="evaluating"):
            wav_id=wav_list[idx]
            scp.append(wav_id)
            wav_path = "{}{}.wav".format(folder, wav_id)
            feature=feature_extract.extract(wav_path, feature_type)
            data=np.reshape(feature, (-1, 1, 45, 600))
            data=torch.Tensor(data).to(device)
            output=model(data)
            output=output[0]
            result.append(output.cpu().numpy().ravel())
        result=np.reshape(result,(-1,2))
        print(result.shape)
        score = (result[:, 1] - result[:, 0])
        with open(name, 'w') as fh:
            for f, cm in zip(scp, score):
                fh.write('{} {}\n'.format(f, cm))
    print('Eval result saved to {}'.format(name))

            
class ScheduledOptim(object):
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 64
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        "Learning rate scheduling per step"

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def state_dict(self):
        ret = {
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_current_steps': self.n_current_steps,
            'delta': self.delta,
        }
        ret['optimizer'] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_current_steps = state_dict['n_current_steps']
        self.delta = state_dict['delta']
        self.optimizer.load_state_dict(state_dict['optimizer'])


def set_seed(seed):

    np.random.seed(seed)   
    torch.manual_seed(seed) 

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch LCNN ASVspoof')

    # parser.add_argument('--phase', dest='phase', default='train', help='train, test')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=True, default='./models_4/')
    
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for dev (default: 16)')
    parser.add_argument('--epochs', type=int, default=32, metavar='N',
                        help='number of epochs to train (default: 9)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--warmup', type=float, default=1000, metavar='M')
    #parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--embedding', default=64, type=int,
                        help='embedding dim of transformer encoder')
    parser.add_argument('--heads', default=1, type=int,
                        help='number of heads of each transformer encoder layer')
    parser.add_argument('--posit', default='sine', type=str,
                        help='type of positional embedding')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--feature_type', default='fft')

    parser.add_argument("--gpu", type=str, help="GPU index", default="1")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("torch.cuda.is_available()", torch.cuda.is_available())
    device = "cuda"
    dtype = torch.float32
    set_seed(args.seed)
    

    if not os.path.exists(args.out_fold):
        os.makedirs(args.out_fold)
    if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
        os.makedirs(os.path.join(args.out_fold, 'checkpoint'))

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 4,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    train_data, train_label =load_data("train", train_protocol, mode="train")
    train_dataset=ASVDataSet(train_data, train_label, mode="train")
    train_dataloader=DataLoader(train_dataset, **kwargs)

    dev_data, dev_label=load_data("dev", dev_protocol, mode="train")
    dev_dataset=ASVDataSet(dev_data, dev_label, mode="train")
    dev_dataloader=DataLoader(dev_dataset, **kwargs)


    mamba_config = MambaConfig()
    model = MambaLMHeadModel(config=mamba_config, device=device, dtype=dtype).to(device)

    optimizer = ScheduledOptim(optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.warmup)
    loss=10
    ploss=1


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_dataloader, optimizer, epoch)
        loss=test(model, device, dev_dataloader)
        
        torch.save(model.state_dict(), os.path.join(args.out_fold, 'checkpoint','senet_epoch_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(args.out_fold, 'checkpoint','op_epoch_%d.pt' % epoch))

        if args.save_model:
            if loss<ploss:
                ploss=loss
                torch.save(model.state_dict(), os.path.join(args.out_fold, 'rawbmamba_best.pt'))
                torch.save(optimizer.state_dict(), os.path.join(args.out_fold, 'op.pt'))
                print("model saved")
    
        



if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    total = round(e - s)
    h = total // 3600
    m = (total-3600*h) // 60
    s = total - 3600*h - 60*m
    print(f"cost time {h}:{m}:{s}")
