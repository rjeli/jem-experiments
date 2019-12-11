#!/usr/bin/env python3
"""
Usage:
    ./fgsm.py <model> <class> <iters> [--energy=<coeff>]
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
    opt = torch.optim.SGD([x], lr=1e-1, weight_decay=0)
    class_idx = classes.index(args['<class>'])
    y = torch.tensor([class_idx]).cuda()
    # for i in range(int(args['<iters>'])):
    while True:
        # print('iter', i)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        print('clf_loss:', loss.item(),
            'y_hat:', [round(p, 2) for p in y_hat[0].cpu().tolist()])
        if args['--energy']:
            energy = -torch.logsumexp(y_hat, dim=1).mean()
            print('energy:', energy.item())
            loss += energy * float(args['--energy'])
            print('final loss:', loss.item())
        if loss.item() < -30:
            print('done!')
            break
        model.zero_grad()
        opt.zero_grad()
        x.retain_grad()
        loss.backward()
        opt.step()

        # sign = torch.sign(x.grad.data)
        # perturbed = x.detach() - 0.01 * sign
        # x = torch.clamp(perturbed, min=-1, max=1)

        print('x norm:', (x**2).sum().item())
        print('grad norm:', (x.grad.data**2).sum().item())

        # x -= 1e-3 * x.grad.data
        # x.grad.zero_()
        # x.requires_grad_(True)

    x_im = TF.to_pil_image(x[0].cpu()*.5+.5)
    x_im.save('fgsm.png')
