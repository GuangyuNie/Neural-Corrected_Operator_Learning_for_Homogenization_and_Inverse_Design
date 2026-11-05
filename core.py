"""Minimal two-dimensional conductivity components, extracted from the research code."""
import torch
from torch import nn

def twopcf(image_tensor):
    assert image_tensor.ndim == 2, 'Input must be a 2D image.'
    F_k = torch.fft.fft2(image_tensor)
    power_spectrum = F_k * torch.conj(F_k)
    correlations = torch.fft.ifft2(power_spectrum).real
    correlations = torch.fft.fftshift(correlations)
    return correlations / image_tensor.numel()

class EffectiveConductivityNPCF:

    def __init__(self, size, sigma0, sigma1, n, d, device, gt_size=None):
        if n != 2 or d != 2:
            raise ValueError('This release supports only d=2 and second-order SCE (n=2).')
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.size = size
        self.n = n
        self.d = d
        self.device = device
        self.gt_size = size
        self.radius_scale = self.gt_size / self.size
        self.microstructure = None
        self.S2 = None
        self.S3 = None
        self.S4 = None
        self.phi = None
        self.A2 = None
        self.A3 = None
        self.A4 = None
        self.D = None
        self.Sigma = None
        self.normalization_factor = 1

    def compute_T(self, r: torch.Tensor, d: int) -> torch.Tensor:
        r_norm = torch.norm(r, dim=1, keepdim=True).clamp_min(1e-08)
        n = r / r_norm
        T = (d * torch.einsum('ij,ik->ijk', n, n) - torch.eye(d, device=r.device).unsqueeze(0)) / (r_norm.unsqueeze(2) * self.radius_scale) ** d
        return T

    def compute_T_with_NN(self, r, model):
        return model(r)

    def compute_beta(self) -> float:
        return (self.sigma1 - self.sigma0) / (self.sigma1 + (self.d - 1) * self.sigma0)

    def compute_phi(self, microstructure=None, store=True) -> float:
        """
        Compute the volume fraction phi (the mean of the microstructure).
        If `microstructure` is None, use `self.microstructure`.
        If `store=True`, store result in self.phi.
        """
        if microstructure is None:
            microstructure = self.microstructure
            if microstructure is None:
                raise ValueError('microstructure not provided or set in class.')
        phi = torch.mean(microstructure)
        if store:
            self.phi = phi
        return phi

    def compute_S2(self, microstructure=None, store=True) -> torch.Tensor:
        """
        Compute the two-point correlation function S2.
        If `microstructure` is None, we use `self.microstructure`.
        If `store=True`, store the result in self.S2.
        """
        if microstructure is None:
            microstructure = self.microstructure
            if microstructure is None:
                raise ValueError('microstructure not provided or set in class.')
        S2 = twopcf(microstructure)
        if store:
            self.S2 = S2
        return S2

    def compute_A2(self, beta: float=None, phi: float=None, S2: torch.Tensor=None, T_marix: torch.Tensor=None, use_NN=False, model=None, delta=None, store=True, get_T=False, get_T_and_A=False) -> torch.Tensor:
        """
        Compute the A2 term, with possible custom S2, phi, etc.
        If none given, fallback to self.S2, self.phi, etc.
        If store=True, save the result to self.A2.
        """
        d = 2
        omega = 2 * torch.pi
        n = 2
        I = J = self.size
        a = torch.repeat_interleave(torch.arange(I), I).view(-1, 1).to(self.device)
        b = torch.arange(J).repeat(J).view(-1, 1).to(self.device)
        indices = torch.cat((b, a), dim=1).to(self.device)
        indices -= int(self.size / 2)
        norms = torch.norm(indices.float(), dim=1)
        valid_mask = norms != 0
        valid_indices = indices[valid_mask]
        valid_indices_ = torch.as_tensor(valid_indices, dtype=torch.float32).requires_grad_(True)
        if T_marix is not None:
            T_matrices = T_marix
        else:
            if not use_NN:
                T_matrices = self.compute_T(valid_indices_, d)
            else:
                if model is None:
                    raise ValueError('use_NN=True but no model provided.')
                T_matrices = self.compute_T_with_NN(valid_indices_, model).real / self.normalization_factor
            if get_T:
                return T_matrices
        if S2 is None:
            if self.S2 is None:
                raise ValueError('S2 not provided and self.S2 is None. Please compute or provide S2.')
            S2 = self.S2
        if beta is None:
            beta = self.compute_beta()
        if phi is None:
            if self.phi is None:
                raise ValueError('phi not provided and self.phi is None. Please compute or provide phi.')
            phi = self.phi
        S = S2.to(self.device)
        valid_S = S.flatten()[valid_mask]
        if delta is None:
            Delta_matrices = (torch.abs(valid_S) - phi ** 2).to(self.device)
        else:
            Delta_matrices = delta.to(self.device)
        A = torch.einsum('ijk,i->jk', T_matrices, Delta_matrices.float())
        A *= (self.radius_scale ** d) ** (n - 1) * d / omega
        if store:
            self.A2 = A
        if get_T_and_A:
            return (A, T_matrices, valid_indices_)
        return A

    def compute_D(self, phi: float=None, A2: torch.Tensor=None, A3: torch.Tensor=None, A4: torch.Tensor=None, store=True) -> torch.Tensor:
        """
        Compute D = beta * phi * I - A2 * beta^2 - A3 * beta^3 - A4 * beta^4
        up to the order n that you specify in self.n.
        If store=True, store result in self.D.
        """
        if phi is None:
            phi = self.phi
        if phi is None:
            raise ValueError('phi is None, cannot compute D. Provide or compute phi first.')
        beta = self.compute_beta()
        d = self.d
        I = torch.eye(d, device=self.device)
        if A2 is None and self.n >= 2:
            if self.A2 is None:
                raise ValueError('A2 not computed or provided.')
            A2 = self.A2
        if A3 is None and self.n >= 3:
            if self.A3 is None:
                raise ValueError('A3 not computed or provided.')
            A3 = self.A3
            test = A3 / A2
        if A4 is None and self.n >= 4:
            if self.A4 is None:
                raise ValueError('A4 not computed or provided.')
            A4 = self.A4
        D = beta * phi * I
        if self.n >= 2 and A2 is not None:
            D -= A2 * beta ** 2
        if self.n >= 3 and A3 is not None:
            D -= A3 * beta ** 3
        if self.n >= 4 and A4 is not None:
            D -= A4 * beta ** 4
        if store:
            self.D = D
        return D

    def compute_Sigma(self, D=None, phi=None, store=True) -> torch.Tensor:
        """
        Compute final Sigma from D:

        Sigma = inverse( (D / (beta^2 (phi+eps)^2)) - I ) * [ (sigma0 / (beta^2 (phi+eps)^2)) * D + (d-1)*sigma0*I ]

        If store=True, store result in self.Sigma.
        """
        eps = 1e-12
        d = self.d
        if D is None:
            D = self.D
        if D is None:
            raise ValueError('D is None, cannot compute Sigma. Provide or compute D first.')
        if phi is None:
            phi = self.phi
        if phi is None:
            raise ValueError('phi is None, cannot compute Sigma. Provide or compute phi first.')
        beta = self.compute_beta()
        I = torch.eye(d, device=self.device)
        Sigma = torch.inverse(D / beta ** 2 / (phi + eps) ** 2 - torch.eye(d, device=self.device)) @ (self.sigma0 / beta ** 2 / (phi + eps) ** 2 * D + (d - 1) * self.sigma0 * torch.eye(d, device=self.device))
        if store:
            self.Sigma = Sigma
        return Sigma

class AnalyticLaplaceT2D(nn.Module):
    """
    Analytic base T for d=2:
      T(r) = (2 n n^T - I) / ( (||r|| * radius_scale)^2 )
    coords: (B,2) in (y,x)
    return: (B,2,2) real (dtype follows coords)
    """

    def __init__(self, radius_scale: float=1.0, eps: float=1e-08):
        super().__init__()
        self.radius_scale = float(radius_scale)
        self.eps = float(eps)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        r = coords
        r_norm = torch.norm(r, dim=1, keepdim=True).clamp_min(self.eps)
        n = r / r_norm
        I = torch.eye(2, device=coords.device, dtype=coords.dtype).unsqueeze(0)
        nnT = torch.einsum('bi,bj->bij', n, n)
        denom = (r_norm * self.radius_scale).pow(2.0)
        T = (2.0 * nnT - I) / denom.unsqueeze(-1)
        return T

class MLPResidual2x2(nn.Module):
    """
    Learns ΔT(coords) directly with an MLP.
    Input coords: (B,2) in (y,x)
    Output: (B,2,2) complex64 (imag=0)
    """

    def __init__(self, hidden: int=64, depth: int=4, eps_r: float=1e-06, use_theta: bool=True, enforce_symmetry: bool=True):
        super().__init__()
        self.eps_r = float(eps_r)
        self.use_theta = bool(use_theta)
        self.enforce_symmetry = bool(enforce_symmetry)
        in_dim = 1 + 2 + (2 if self.use_theta else 0)
        layers = []
        d_in = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        layers += [nn.Linear(hidden, 4)]
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        y = coords[:, 0]
        x = coords[:, 1]
        r = torch.sqrt(x * x + y * y + self.eps_r)
        nx = x / r
        ny = y / r
        feats = [torch.log(r), nx, ny]
        if self.use_theta:
            theta = torch.atan2(y, x)
            feats += [torch.cos(theta), torch.sin(theta)]
        inp = torch.stack(feats, dim=1)
        out = self.net(inp)
        out = out.view(-1, 2, 2)
        if self.enforce_symmetry:
            out01 = 0.5 * (out[:, 0, 1] + out[:, 1, 0])
            out = out.clone()
            out[:, 0, 1] = out01
            out[:, 1, 0] = out01
        return out.to(torch.complex64)
