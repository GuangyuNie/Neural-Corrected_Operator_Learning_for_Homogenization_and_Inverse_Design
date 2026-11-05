"""Neural-corrected second-order SCE for two-dimensional conductivity."""
import torch
from torch import nn

from core import AnalyticLaplaceT2D, EffectiveConductivityNPCF, MLPResidual2x2


class NCO(nn.Module):
    """Shared coordinate kernel; material parameters enter the SCE closure.

    twopcf: centered [batch, 1, size, size] two-point correlations.
    sigma0, sigma1: phase conductivities, each scalar or [batch].
    phi: optional [batch] volume fractions. The default uses S2(0), which
         equals phi for binary indicators, but not for continuous densities.
    All tensor components and lag coordinates use the same x/y ordering.
    """

    def __init__(self, size, hidden=64, depth=4):
        super().__init__()
        if size < 4 or size % 2:
            raise ValueError('Use an even grid size of at least 4.')
        self.size = int(size)
        self.base = AnalyticLaplaceT2D(radius_scale=1.0)
        self.mlp = MLPResidual2x2(hidden=hidden, depth=depth, enforce_symmetry=False)
        lag = torch.arange(size, dtype=torch.float32) - size // 2
        yy, xx = torch.meshgrid(lag, lag, indexing='ij')
        offsets = torch.stack((xx.flatten(), yy.flatten()), dim=1)
        self.register_buffer('offsets', offsets[offsets.square().sum(dim=1) != 0])

    def kernel(self, offsets):
        """T_hat = T0 + T0 * tanh(f_theta), with componentwise multiplication.

        T = 2*pi*sigma0*H in the conductivity SCE normalization. Therefore
        the same componentwise modulation applies to the physical H kernel.
        The singular zero-displacement entry must not be supplied.
        """
        if torch.any(offsets.square().sum(dim=1) == 0):
            raise ValueError('The nonlocal kernel excludes zero displacement.')
        analytical = self.base(offsets)
        return analytical * (1.0 + torch.tanh(self.mlp(offsets).real))

    def structural_loss(self, offsets=None):
        """Soft Hessian-compatibility and tensor-symmetry penalties."""
        coords = self.offsets if offsets is None else offsets
        coords = coords.detach().clone().requires_grad_(True)
        kernel = self.kernel(coords)
        gradients = [
            torch.autograd.grad(kernel[:, i, j].sum(), coords, create_graph=True,
                                retain_graph=True)[0]
            for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))
        ]
        d00, d01, d10, d11 = gradients
        return ((d00[:, 1] - d01[:, 0]).square().mean()
                + (d10[:, 1] - d11[:, 0]).square().mean()
                + (kernel[:, 0, 1] - kernel[:, 1, 0]).square().mean())

    def forward(self, twopcf, sigma0, sigma1, phi=None):
        if twopcf.ndim != 4 or twopcf.shape[1:] != (1, self.size, self.size):
            raise ValueError('twopcf must have shape [batch, 1, size, size].')
        if twopcf.dtype != torch.float32:
            raise ValueError('This extracted implementation uses float32.')
        batch = twopcf.shape[0]
        s0 = torch.as_tensor(sigma0, device=twopcf.device).reshape(-1).expand(batch)
        s1 = torch.as_tensor(sigma1, device=twopcf.device).reshape(-1).expand(batch)
        if torch.any(s0 <= 0) or torch.any(s1 <= 0) or torch.any(s0 == s1):
            raise ValueError('Use positive, distinct phase conductivities.')
        if phi is None:
            phi = twopcf[:, 0, self.size // 2, self.size // 2]
        else:
            phi = torch.as_tensor(phi, device=twopcf.device,
                                  dtype=twopcf.dtype).reshape(-1).expand(batch)
        if torch.any(phi <= 0) or torch.any(phi > 1):
            raise ValueError('The SCE closure requires 0 < phi <= 1.')
        kernel = self.kernel(self.offsets)
        predictions = []
        for i in range(batch):
            sce = EffectiveConductivityNPCF(self.size, float(s0[i]), float(s1[i]),
                                           2, 2, twopcf.device)
            a2 = sce.compute_A2(S2=twopcf[i, 0], phi=phi[i],
                                T_marix=kernel, store=False)
            d = sce.compute_D(phi=phi[i], A2=a2, store=False)
            predictions.append(sce.compute_Sigma(D=d, phi=phi[i], store=False))
        return torch.stack(predictions)
