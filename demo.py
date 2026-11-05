"""Generate conductivity labels, train NCO, and differentiate with respect to S2."""
import argparse
from pathlib import Path

import numpy as np
import torch

from core import twopcf
from nco import NCO
from reference import effective_conductivity_pde, generate_correlated_random_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=Path('runs/nco.pt'))
    args = parser.parse_args()
    if args.samples < 2 or args.steps < 1:
        parser.error('Use at least two samples and one training step.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = NCO(args.size).to(device)
    correlations, targets, s0s, s1s = [], [], [], []
    for i in range(args.samples):
        phi = (0.2, 0.3, 0.4)[i % 3]
        s0, s1 = (4.0, 5.0, 6.0)[i % 3], (40.0, 50.0, 60.0)[(i // 3) % 3]
        material, _ = generate_correlated_random_field(
            size=args.size, mean=phi,
            length_scale_x=(0.04, 0.06, 0.08)[(i // 3) % 3],
            length_scale_y=0.02, seed=args.seed + i)
        correlations.append(twopcf(torch.tensor(material, dtype=torch.float32)))
        targets.append(effective_conductivity_pde(material, s0, s1).compute())
        s0s.append(s0)
        s1s.append(s1)
    inputs = torch.stack(correlations).unsqueeze(1).to(device)
    target = torch.tensor(np.stack(targets), dtype=torch.float32, device=device)
    s0 = torch.tensor(s0s, device=device)
    s1 = torch.tensor(s1s, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    regularization_offsets = model.offsets[::max(1, len(model.offsets) // 64)]
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad()
        prediction = model(inputs, s0, s1)
        data_loss = (prediction - target).square().mean()
        loss = data_loss + 1e-3 * model.structural_loss(regularization_offsets)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss; inspect the SCE closure and material contrast.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or step == args.steps - 1:
            print(f'Step {step + 1}: training tensor MSE = {data_loss.item():.6f}')
    model.eval()
    sample = inputs[:1].detach().clone().requires_grad_(True)
    # Fix phi when taking the descriptor-space partial derivative.
    fixed_phi = sample[:, 0, args.size // 2, args.size // 2].detach()
    sigma_xx = model(sample, s0[:1], s1[:1], phi=fixed_phi)[0, 0, 0]
    sensitivity, = torch.autograd.grad(sigma_xx, sample)
    print(f'sigma_xx = {sigma_xx.item():.6f}; ||d sigma_xx / d S2|| = {sensitivity.norm().item():.6f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'size': args.size, 'state_dict': model.state_dict()}, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
