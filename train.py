#!/usr/bin/env python3
"""
Usage:
    ./train.py --epochs=<n> [--samples=<n>] [--bs=<n>] --save-to=<path> [--jem]
"""

from tqdm import tqdm
from docopt import docopt
import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1
from torchvision.datasets import CIFAR10
import torchvision.transforms as tfm
from torch.utils.data import DataLoader

img_norm = tfm.Normalize((.5,.5,.5),(.5,.5,.5))
img_tfm = tfm.Compose([tfm.ToTensor(), img_norm])

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.in_planes = 16
        self.layers = nn.Sequential(
            self._make_layer(16, 3, stride=1),
            self._make_layer(32, 3, stride=2),
            self._make_layer(64, 3, stride=2),
        )
        self.linear = nn.Linear(64, 10)
    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            downsample = None
            if self.in_planes != planes or s != 1:
                downsample = conv1x1(
                    self.in_planes, planes*BasicBlock.expansion, stride=s)
            layers.append(BasicBlock(self.in_planes, planes, 
                stride=s, downsample=downsample, norm_layer=nn.Identity))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

class JEM:
    def __init__(self, model):
        self.model = model
        self.replay_buffer = collections.deque(maxlen=10000)
    def energy_at(self, x):
        #return -self.model(x).clamp(max=10).exp().sum(dim=1).clamp(min=1e-4).log()
        return -torch.logsumexp(self.model(x), dim=1)
    def draw_sample(self):
        if self.replay_buffer and torch.rand(1) < .95:
            x = random.choice(self.replay_buffer)
        else:
            x = torch.rand(3, 32, 32).cuda() * 2 - 1
        for _ in range(20):
            x.requires_grad_(True)
            energy = self.energy_at(x[None])[0]
            energy.backward()
            x_grad = x.grad.clone()
            x.requires_grad_(False)
            x = x - x_grad + .01 * torch.randn_like(x)
        self.replay_buffer.append(x)
        print(f'got sample with energy {energy.item()}')
        return x
    def gen_loss(self, x):
        sampled_x = self.draw_sample()
        # ?
        # return F.mse_loss(self.energy_at(x), self.energy_at(sampled_x[None]))
        return self.energy_at(x) - self.energy_at(sampled_x[None])

if __name__ == '__main__':
    args = docopt(__doc__)

    print('loading resnet')
    model = Resnet().cuda()
    print('done')

    x = torch.randn(1, 3, 32, 32).cuda()
    y = model(x)
    print(y.shape)

    bs = int(args['--bs'] or 32)
    if args['--jem']:
        jem = JEM(model)
        bs = 1

    tds = CIFAR10('~/cifar', train=True, download=True, transform=img_tfm)
    vds = CIFAR10('~/cifar', train=False, download=True, transform=img_tfm)
    if args['--samples']:
        samples = int(args['--samples'])
        tds = torch.utils.data.Subset(tds, list(range(samples)))
        vds = torch.utils.data.Subset(vds, list(range(samples)))
    print('len(tds):', len(tds))
    print('len(vds):', len(tds))

    tdl = DataLoader(tds, 
        shuffle=True, batch_size=bs, num_workers=8, pin_memory=True)
    vdl = DataLoader(vds, 
        shuffle=False, batch_size=bs, num_workers=8, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), 
        # no batchnorm, so make sure we weight_decay
        lr=1e-3, weight_decay=1e-2)

    for epoch in range(int(args['--epochs'])):
        print('epoch', epoch)
        model.train()
        train_loss = 0.
        for img, y in tqdm(tdl):
            img, y = img.cuda(), y.cuda()
            y_hat = model(img)
            loss = F.cross_entropy(y_hat, y)
            if args['--jem']:
                gen_loss = jem.gen_loss(img)
                print('clf_loss:', loss.item())
                print('gen_loss:', gen_loss.item())
                loss += gen_loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        print('train loss:', train_loss/len(tdl))
        model.eval()
        val_loss = 0.
        n_correct = 0
        conf_counts = [0] * 20
        conf_corrects = [0] * 20
        with torch.no_grad():
            for img, y in tqdm(vdl):
                y_hat = model(img.cuda())
                loss = F.cross_entropy(y_hat, y.cuda())
                val_loss += loss.item()
                n_correct += torch.sum(y_hat.cpu().max(dim=1)[1] == y).item()
                confs = F.softmax(y_hat, dim=1)
                for batch_i in range(confs.size()[0]):
                    _, pred_cls = confs[batch_i].max(dim=0)
                    for cls_i in range(confs.size()[1]):
                        conf = confs[batch_i, cls_i].clamp(min=0,max=.99)
                        bucket = int(conf * 20)
                        conf_counts[bucket] += 1
                        if cls_i == pred_cls and cls_i == y[batch_i]:
                            conf_corrects[bucket] += 1
        print('val loss:', val_loss/len(vds))
        print('val accuracy:', n_correct/len(vds))
        calib_err = 0.
        for i in range(20):
            if conf_counts[i]:
                model_density = conf_corrects[i] / conf_counts[i]
                calib_err += abs(i/20 - model_density)
            else:
                model_density = 0
            print(f'bucket {round(i/20,2)}: {round(model_density,2)}')
        print('val calib err:', calib_err)

    print('saving to', args['--save-to'])
    torch.save(model.state_dict(), args['--save-to'])
    print('done')

