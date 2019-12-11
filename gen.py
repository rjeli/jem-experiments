#!/usr/bin/env python3
"""
Usage:
    ./gen.py <model> <class> <stop>
"""

from docopt import docopt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from train import Resnet, img_norm

classes = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]

if __name__ == '__main__':
    args = docopt(__doc__)
    model = Resnet().cuda()
    model.load_state_dict(torch.load(args['<model>'])['model'])
    model.train()
    x = torch.randn(1, 3, 32, 32, requires_grad=True, device='cuda')
    opt = torch.optim.SGD([x], lr=1e-1, weight_decay=1e-5)
    class_idx = classes.index(args['<class>'])
    y = torch.tensor([class_idx]).cuda()
    i = 0
    while True:
        print('iter', i)
        i += 1
        y_hat = model(x)
        print('y_hat:', [round(p, 2) for p in y_hat[0].cpu().tolist()])
        energy = -torch.logsumexp(y_hat, dim=1).mean()
        print('energy:', energy.item())
        clf_loss = F.cross_entropy(y_hat, y)
        print('clf_loss:', clf_loss.item())
        loss = -y_hat[0,class_idx]
        print('loss:', loss.item())
        if loss.item() < -int(args['<stop>']):
            print('done!')
            break

        model.zero_grad()
        opt.zero_grad()
        x.retain_grad()
        loss.backward()
        opt.step()

        print('x norm:', (x**2).sum().item())
        print('grad norm:', (x.grad.data**2).sum().item())
        print('x:', x.min().item(), x.max().item(), x.std().item())
        
        with torch.no_grad():
            x.clamp_(min=-1, max=1)

    x_im = TF.to_pil_image(x[0].cpu()*.5+.5)
    x_im.save('sample.png')
