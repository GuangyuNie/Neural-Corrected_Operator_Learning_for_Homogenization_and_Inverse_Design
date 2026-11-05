"""Binary 2D conductivity reference solver and correlated microstructure generator."""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

class effective_conductivity_pde:

    def __init__(self, microstructure, sigma0, sigma1):
        self.ms = microstructure.astype(np.int8)
        self.n = self.ms.shape[0]
        assert self.ms.shape[0] == self.ms.shape[1], '需要正方网格'
        self.s0 = float(sigma0)
        self.s1 = float(sigma1)
        self.L = self.n - 1

    def sigma_field(self):
        return np.where(self.ms == 0, self.s0, self.s1).astype(float)

    @staticmethod
    def _harmonic(a, b, eps=1e-30):
        return 2.0 * a * b / (a + b + eps)

    def _is_dirichlet(self, i, j, drive):
        n = self.n
        if drive == 'x':
            return j == 0 or j == n - 1
        elif drive == 'y':
            return i == 0 or i == n - 1
        else:
            raise ValueError("drive must be 'x' or 'y'")

    def _phi_dirichlet_value(self, i, j, drive):
        if drive == 'x':
            return 1.0 - j / self.L
        else:
            return 1.0 - i / self.L

    def _assemble_A_b_mixed(self, sigma, drive='x'):
        """
        混合边界：
          drive='x'：左右 Dirichlet（线性势差），上下 Neumann（无通量）
          drive='y'：上下 Dirichlet（线性势差），左右 Neumann（无通量）
        """
        n = self.n
        idx = lambda i, j: i * n + j
        A = lil_matrix((n * n, n * n))
        b = np.zeros(n * n)
        for i in range(n):
            for j in range(n):
                p = idx(i, j)
                if self._is_dirichlet(i, j, drive):
                    A[p, p] = 1.0
                    b[p] = self._phi_dirichlet_value(i, j, drive)
                    continue
                diag = 0.0
                if i - 1 >= 0:
                    sN = self._harmonic(sigma[i, j], sigma[i - 1, j])
                    A[p, idx(i - 1, j)] = -sN
                    diag += sN
                else:
                    pass
                if i + 1 < n:
                    sS = self._harmonic(sigma[i, j], sigma[i + 1, j])
                    A[p, idx(i + 1, j)] = -sS
                    diag += sS
                else:
                    pass
                if j - 1 >= 0:
                    sW = self._harmonic(sigma[i, j], sigma[i, j - 1])
                    A[p, idx(i, j - 1)] = -sW
                    diag += sW
                else:
                    pass
                if j + 1 < n:
                    sE = self._harmonic(sigma[i, j], sigma[i, j + 1])
                    A[p, idx(i, j + 1)] = -sE
                    diag += sE
                else:
                    pass
                A[p, p] = diag
        return (A.tocsr(), b)

    def _solve_phi(self, A, b):
        phi = spsolve(A, b)
        return phi.reshape(self.n, self.n)

    def _flux_faces(self, sigma, phi):
        """
        面通量（与离散一致）：
          Jx(i, j+1/2) = -sigma_e * (phi[i, j+1] - phi[i, j])
          Jy(i+1/2, j) = -sigma_s * (phi[i+1, j] - phi[i, j])
        """
        sig_e = self._harmonic(sigma[:, :-1], sigma[:, 1:])
        Jx = -sig_e * (phi[:, 1:] - phi[:, :-1])
        sig_s = self._harmonic(sigma[:-1, :], sigma[1:, :])
        Jy = -sig_s * (phi[1:, :] - phi[:-1, :])
        return (Jx, Jy)

    def compute(self):
        sigma = self.sigma_field()
        A_x, b_x = self._assemble_A_b_mixed(sigma, drive='x')
        phi_x = self._solve_phi(A_x, b_x)
        Jx_faces_x, Jy_faces_x = self._flux_faces(sigma, phi_x)
        A_y, b_y = self._assemble_A_b_mixed(sigma, drive='y')
        phi_y = self._solve_phi(A_y, b_y)
        Jx_faces_y, Jy_faces_y = self._flux_faces(sigma, phi_y)
        E0 = 1.0 / self.L
        Jx_x = Jx_faces_x.mean()
        Jy_x = Jy_faces_x.mean()
        Jx_y = Jx_faces_y.mean()
        Jy_y = Jy_faces_y.mean()
        Sigma = np.array([[Jx_x / E0, Jx_y / E0], [Jy_x / E0, Jy_y / E0]])
        return Sigma

def generate_correlated_random_field(size, mean, length_scale_x, length_scale_y, seed=None):
    """
    Generates a square 2D spatially correlated random field and binarizes it.

    Parameters:
    - size (int): The dimensions of the square random field (size x size).
    - mean (float): The mean value of the random field.
    - cov (float): The variance of the Gaussian random field.
    - correlation_length (float): The correlation length, controlling spatial correlation.
    - threshold (float, optional): The threshold for binarization.
                                   If None, the mean of the generated field will be used.

    Returns:
    - binary_field (np.ndarray): The binarized 2D random field.
    - random_field (np.ndarray): The generated random field before binarization.
    """
    nx = size
    ny = size
    if seed != None:
        np.random.seed(seed=seed)
    kx = np.fft.fftfreq(nx, 1.0 / nx)
    ky = np.fft.fftfreq(ny, 1.0 / ny)
    kx, ky = np.meshgrid(kx, ky)
    power_spectrum = np.exp(-0.5 * ((kx * length_scale_x) ** 2 + (ky * length_scale_y) ** 2))
    random_complex = np.random.normal(0, 1, (nx, ny)) + 1j * np.random.normal(0, 1, (nx, ny))
    random_field = np.real(np.fft.ifft2(np.fft.fft2(random_complex) * np.sqrt(power_spectrum)))
    random_field = random_field - np.mean(random_field)
    random_field = random_field / np.std(random_field)
    random_field = random_field + mean
    flat_vals = random_field.ravel()
    T = np.quantile(flat_vals, 1 - mean)
    binary_field = (random_field > T).astype(float)
    return (binary_field, random_field)
