import torch
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply

from utils import timer, energy_of_pred

class SGLDSampler:
    def __init__(self):
        self.ngpus = torch.cuda.device_count()
        print('have', self.ngpus, 'gpus')
    def draw(self, model, num_samples, num_steps=80):
        assert num_samples % self.ngpus == 0, 'batch size must be divisible'
        gpus = list(range(self.ngpus))
        with timer('genxs'):
            xs = [torch.rand(num_samples//self.ngpus, 3, 32, 32, device=i) * 2 - 1
                  for i in gpus]
            for x in xs:
                x.requires_grad_(True)
        with timer('replicate model'):
            model.eval()
            models = replicate(model, gpus, detach=True)
            model.train()
        energy_befores = [m.energy(x).mean().item() for m, x in zip(models, xs)]
        print('energy befores:', energy_befores)
        for _ in range(num_steps):
            preds = parallel_apply(models, xs, devices=gpus)
            energies = [energy_of_pred(p).sum() for p in preds]
            g = torch.autograd.grad(energies, xs, retain_graph=True)
            # print('norms:', [gg.norm(2) for gg in g])
            for i in gpus:
                xs[i].data.add_(-1, g[i])
                xs[i].data.add_(.01, torch.randn_like(xs[i]))
        energy_afters = [m.energy(x).mean().item() for m, x in zip(models, xs)]
        print('energy afters:', energy_afters)
        return torch.cuda.comm.gather(xs)

