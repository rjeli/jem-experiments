#!/usr/bin/env python3
"""
Usage:
    ./train.py --epochs=<n> [--samples=<n>] [--bs=<n>] --save-to=<path> [--jem] [--resume=<path>] [--seed=<n>] [--lr=<lr>] [--sgld-steps=<n>]
"""

from tqdm import tqdm
from docopt import docopt
import collections
import time
from pathlib import Path
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1
from torchvision.datasets import CIFAR10
import torchvision.transforms as tfm
from torch.utils.data import DataLoader
import visdom

img_norm = tfm.Normalize((.5,.5,.5),(.5,.5,.5))
img_tfm = tfm.Compose([tfm.ToTensor(), img_norm])

NGPUS = torch.cuda.device_count()

from sgld import SGLDSampler
from utils import timer, energy_of_pred

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
        x = F.avg_pool2d(x, x.shape[3])
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
    def energy(self, x):
        return energy_of_pred(self.forward(x))

class VisStats:
    def __init__(self):
        # self.vis = visdom.Visdom()
        self.stats = collections.defaultdict(lambda: [])
        self.running = collections.defaultdict(lambda: [])
    def add(self, name, val):
        self.running[name].append(val)
    def update(self):
        for k in sorted(self.running.keys()):
            avg = np.mean(self.running[k])
            del self.running[k]
            print(f'{k}: {avg}')
            self.stats[k].append(avg)
            # self.vis.line(self.stats[k], np.arange(len(self.stats[k])),
                # win=k, opts=dict(title=k))
    def reset(self):
        self.running.clear()

class CalibErrTracker:
    def __init__(self):
        self.conf_counts = [0] * 20
        self.conf_corrects = [0] * 20
    def update(self, y_hat):
        confs = F.softmax(y_hat, dim=1)
        for batch_i in range(confs.size()[0]):
            _, pred_cls = confs[batch_i].max(dim=0)
            for cls_i in range(confs.size()[1]):
                conf = confs[batch_i, cls_i].clamp(min=0,max=.99)
                bucket = int(conf * 20)
                self.conf_counts[bucket] += 1
                if cls_i == pred_cls and cls_i == y[batch_i]:
                    self.conf_corrects[bucket] += 1
    def err(self):
        calib_err = 0.
        for i in range(20):
            if self.conf_counts[i]:
                model_density = self.conf_corrects[i] / self.conf_counts[i]
                calib_err += abs(i/20 - model_density)
        return calib_err

class JEM:
    def __init__(self, model, steps=20):
        self.model = model
        self.replay_buffer = collections.deque(maxlen=10000)
        self.steps = steps
    def reset(self):
        self.replay_buffer.clear()
    def energy_at(self, x):
        return -torch.logsumexp(self.model(x), dim=1)
    def draw_samples(self, num=1):
        xs = []
        for _ in range(num):
            if False and self.replay_buffer and torch.rand(1) < .95:
                rand_idx = (torch.rand(1)*len(self.replay_buffer)).int().item()
                xs.append(self.replay_buffer[rand_idx])
            else:
                xs.append(torch.rand(3, 32, 32).cuda() * 2 - 1)
        xs = torch.stack(xs)
        # if replay buffer is cold, run more steps
        num_steps = 80 if not self.replay_buffer else self.steps
        energy_before = self.energy_at(xs).mean().item()
        print('energy before:', energy_before)
        for _ in range(num_steps):
            xs.requires_grad_(True)
            energy = self.energy_at(xs)
            self.model.zero_grad()
            energy.sum().backward()
            x_grad = xs.grad.clone()
            xs.requires_grad_(False)
            xs = xs - x_grad + .01 * torch.randn_like(xs)
        print('energy after:', self.energy_at(xs).mean().item())
        for x in xs:
            self.replay_buffer.append(x.clone())
        energies = energy.detach().cpu().numpy().tolist()
        print(f'got samples with energies {[round(e, 2) for e in energies]}')
        return xs
    def gen_loss(self, x):
        sampled_xs = self.draw_samples(num=x.shape[0])
        sampled_energies = self.energy_at(sampled_xs)
        # sometimes energy is huge, just ignore it?
        # (the paper said this wouldn't work)
        if False:
            mask = sampled_energies < 0
            return (self.energy_at(x) - sampled_energies)[mask].mean()
        return (self.energy_at(x) - sampled_energies).mean()

if __name__ == '__main__':
    print('training!!')
    args = docopt(__doc__)
    vis = VisStats()
    if args['--seed']:
        torch.manual_seed(int(args['--seed']))
    else:
        torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    for i in range(torch.cuda.device_count()):
        torch.rand(i+1, device=i)

    save_dir = Path(args['--save-to'])
    if not save_dir.exists():
        save_dir.mkdir()

    print('loading resnet')
    model = Resnet().cuda()
    print('done')

    x = torch.randn(1, 3, 32, 32).cuda()
    y = model(x)
    print(y.shape)

    bs = int(args['--bs'] or 32)
    if args['--jem']:
        # bs = 1
        pass

    tds = CIFAR10('~/cifar', train=True, download=True, transform=img_tfm)
    vds = CIFAR10('~/cifar', train=False, download=True, transform=img_tfm)
    if args['--samples']:
        samples = int(args['--samples'])
        tds = torch.utils.data.Subset(tds, list(range(samples)))
        vds = torch.utils.data.Subset(vds, list(range(samples)))
    print('len(tds):', len(tds))
    print('len(vds):', len(tds))

    tdl = DataLoader(tds,
        shuffle=True, batch_size=bs, num_workers=0, pin_memory=True)
    vdl = DataLoader(vds,
        shuffle=False, batch_size=bs, num_workers=0, pin_memory=True)

    lr = float(args['--lr'] or 1e-3)
    opt = torch.optim.AdamW(model.parameters(),
        # no batchnorm, so make sure we weight_decay
        lr=lr, weight_decay=1e-1)

    if args['--resume']:
        print('resuming from', args['--resume'])
        ckpt = torch.load(args['--resume'])
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])

    jem = JEM(model, int(args['--sgld-steps'] or 20))

    sgld = SGLDSampler()

    clf_losses = []
    gen_losses = []

    epoch = 0
    num_restarts = 0
    already_went_back_2 = False
    # for epoch in range(int(args['--epochs'])):
    while epoch < int(args['--epochs']):
        print('epoch', epoch)
        model.train()
        train_energies = []
        cet = CalibErrTracker()
        restarting = False
        for img, y in tqdm(tdl):
            img, y = img.cuda(non_blocking=True), y.cuda(non_blocking=True)
            y_hat = model(img)
            cet.update(y_hat.cpu())
            # train_energies.append(jem.energy_at(img).mean().item())
            train_energies.append(0)
            if train_energies[-1] > 1e3:
                print('=== DETECTED DIVERGENCE, reseeding and restarting ===')
                if epoch > 0:
                    load_path = save_dir / f'{epoch-1}.pt'
                else:
                    load_path = args['--resume']
                current_lr = opt.param_groups[0]['lr']
                if num_restarts > 10:
                    if not already_went_back_2:
                        print('=== TOO MANY RESTARTS, going back 2 epochs ===')
                        load_path = save_dir / f'{epoch-2}.pt'
                        already_went_back_2 = True
                    else:
                        print('=== THAT DIDNT WORK, lowering lr ===')
                        new_lr = current_lr / 10
                        print(f'{current_lr} -> {new_lr}')
                        already_went_back_2 = False
                    num_restarts = 0
                else:
                    new_lr = current_lr
                vis.reset()
                # jem.reset()
                ckpt = torch.load(str(load_path))
                model.load_state_dict(ckpt['model'])
                opt.load_state_dict(ckpt['opt'])
                for p in opt.param_groups:
                    p['lr'] = new_lr
                torch.manual_seed(random.randint(0, 123456))
                num_restarts += 1
                restarting = True
                break
            clf_loss = F.cross_entropy(y_hat, y)
            vis.add('train_clf_loss', clf_loss.item())
            acc = (y_hat.max(dim=1)[1] == y).cpu().float().mean()
            vis.add('train_accuracy', acc.item())
            loss = clf_loss
            """
            if args['--jem']:
                torch.cuda.synchronize()
                with timer('gen_loss'):
                    gen_loss = jem.gen_loss(img)
                    torch.cuda.synchronize()
                print('gen_loss:', gen_loss.item())
                vis.add('train_gen_loss', gen_loss.item())
                loss += 1 * gen_loss.mean()
                # print('energies:', jem.energy_at(img).detach().cpu().tolist())
            """
            if args['--jem']:
                gen_loss = model.energy(img).mean()
                print('drawing sgld')
                post_xs = sgld.draw(model, img.shape[0])
                gen_loss -= model.energy(post_xs).mean()
                loss += gen_loss
            opt.zero_grad()
            loss.backward()
            total_norm = 0
            for p in model.parameters():
                total_norm += (p.grad.data**2).sum().item()
            vis.add('grad_norm', total_norm)
            opt.step()
        if restarting:
            continue
        vis.add('train_calib_err', cet.err())
        model.eval()
        val_energies = []
        cet = CalibErrTracker()
        with torch.no_grad():
            for img, y in tqdm(vdl):
                img, y = img.cuda(), y.cuda()
                y_hat = model(img)
                cet.update(y_hat.cpu())
                val_energies.append(jem.energy_at(img).mean().item())
                clf_loss = F.cross_entropy(y_hat, y)
                vis.add('val_clf_loss', clf_loss.item())
                loss = clf_loss
                acc = (y_hat.max(dim=1)[1] == y).cpu().float().mean()
                vis.add('val_accuracy', acc.item())
        vis.add('val_calib_err', cet.err())
        train_energy = np.mean(train_energies)
        val_energy = np.mean(val_energies)
        rand_img = torch.randn(64, 3, 224, 224).cuda() * 2 - 1
        rand_energy = jem.energy_at(rand_img).mean().item()
        vis.add('log(train_x_prob/rand_x_prob)', rand_energy-train_energy)
        vis.add('log(val_x_prob/rand_x_prob)', rand_energy-val_energy)

        vis.update()

        if vis.stats['train_clf_loss'][-1] > 10:
            print('diverged, bailing out!')
            import sys; sys.exit()

        save_path = save_dir / f'{epoch}.pt'
        print('saving to', save_path)
        torch.save({
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }, str(save_path))
        print('done')

        epoch += 1

    print('done all')
