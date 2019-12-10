#!/usr/bin/env python3
"""
Usage:
    ./fgsm.py <model> <class> <iters>
"""

from docopt import docopt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from train import Resnet, img_norm

if __name__ == '__main__':
    args = docopt(__doc__)
    model = Resnet().cuda()
    model.load_state_dict(torch.load(args['<model>']))
    model.train()
    x = torch.randn(1, 3, 32, 32, requires_grad=True).cuda()
    y = torch.tensor([int(args['<class>'])]).cuda()
    for i in range(int(args['<iters>'])):
        print('iter', i)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        print('loss:', loss.item(), 
            'y_hat:', [round(p, 2) for p in y_hat[0].cpu().tolist()])
        model.zero_grad()
        x.retain_grad()
        loss.backward()

        sign = torch.sign(x.grad.data)
        perturbed = x.detach() - 0.1 * sign
        x = torch.clamp(perturbed, min=-1, max=1)
        x.requires_grad_(True)

    x_im = TF.to_pil_image(x[0].cpu()*.25+.5)
    x_im.save('fgsm.png')

