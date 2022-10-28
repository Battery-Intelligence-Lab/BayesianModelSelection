import torch
from ._gaussian_calc import GaussianCalc
from torch.distributions.multivariate_normal import MultivariateNormal


class SquareRootAcquisitionFunction(GaussianCalc):
    def __init__(self, prior, model, device, n_gaussians=100, threshold=1e-5):
        """
        Inherited the functions from GaussianCalc.

        Args:
        - prior: torch.distributions, prior distribution
        - device: torch.device, cpu or cuda
        """
        super().__init__(prior, device)
        self.n_gaussians = n_gaussians    # number of Gaussians for uncertainty sampling
        self.threshold = threshold        # threshold to cut off the small weights
        self.update(model)

    def update(self, model):
        """
        Args:
        - model: gpytorch.models, function of GP model, typically model = self.gp.model in _basq.py
        """
        self.parameters_extraction(model)
        self.wA, self.wAA, self.mu_AA, self.sigma_AA = self.sparseGMM()
        self.d_AA = len(self.mu_AA)
        self.w_mean, self.mu_mean, self.sig_mean = self.sparseGMM_mean()
        self.d_mean = len(self.mu_mean)

    def sparseGMM(self):
        """
        See details on factorisation trick and sparse GMM sampler in Sapplementary.
        https://arxiv.org/abs/2206.04734

        Returns:
            - w1: torch.tensor, the weight of prior distribution
            - w2: torch.tensor, the weights of other normal distributions
            - mu2: torch.tensor, the mean vectors of other normal distributions
            - sigma2: torch.tensor, the covariance matrix of other normal distributions
        """
        i, j = torch.where(self.woodbury_inv < 0)
        _w1_ = self.outputscale
        _w2_ = torch.abs((self.v**2) * self.woodbury_inv[i, j])
        _Z = _w1_ + torch.sum(_w2_)
        _w1, _w2 = _w1_ / _Z, _w2_ / _Z

        Winv = self.W.inverse()
        Sinv = self.prior.covariance_matrix.inverse()
        sigma2 = (2 * Winv + Sinv).inverse()

        _idx = _w2.argsort(descending=True)[:self.n_gaussians]
        idx = _idx[_w2[_idx] > self.threshold]
        Xi = self.Xobs[i[idx]]
        Xj = self.Xobs[j[idx]]

        w2 = _w2[idx]
        mu2 = (sigma2 @ Winv @ (Xi + Xj).T).T + sigma2 @ Sinv @ self.prior.loc

        zA = _w1 + torch.sum(w2)
        w1, w2 = _w1 / zA, w2 / zA
        return w1, w2, mu2, sigma2

    def joint_pdf(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - first/first+second: torch.tensor, the values of probability density function of approximated A(x)
        """
        d_x = len(x)

        # calculate the first term
        Npdfs_A = self.utils.safe_mvn_prob(
            self.prior.loc,
            self.prior.covariance_matrix,
            x,
        )
        first = self.wA * Npdfs_A

        # calulate the second term
        if len(self.wAA) == 0:
            return first
        else:
            x_AA = (torch.tile(self.mu_AA, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
                self.d_AA * d_x, self.n_dims
            )
            Npdfs_AA = self.utils.safe_mvn_prob(
                torch.zeros(self.n_dims).to(self.device),
                self.sigma_AA,
                x_AA,
            ).reshape(d_x, self.d_AA)

            f_AA = self.wAA.unsqueeze(0) * Npdfs_AA
            second = f_AA.sum(axis=1)
            return first + second

    def sampling(self, n):
        """
        Args:
            - n: int, number of samples to be sampled.

        Returns:
            - samplesA/samplesAA: torch.tensor, the samples from approximated A(x)
        """
        cntA = (n * self.wA).type(torch.int)
        samplesA = self.prior.sample(torch.Size([cntA])).to(self.device)

        if len(self.wAA) == 0:
            return samplesA
        else:
            cntAA = (n * self.wAA).type(torch.int)
            samplesAA = torch.cat([
                MultivariateNormal(
                    self.mu_AA[i],
                    self.sigma_AA,
                ).sample(torch.Size([cnt])).to(self.device)
                for i, cnt in enumerate(cntAA)
            ])
            return torch.cat([samplesA, samplesAA])

    def sparseGMM_mean(self):
        """
        Returns:
            - weights: torch.tensor, the weight of approximated GP mean functions
            - mu_mean: torch.tensor, the mean vectors of approximated GP mean functions
            - sig_prime: torch.tensor, the covariance matrix of approximated GP mean functions
        """
        Winv = self.W.inverse()
        Sinv = self.prior.covariance_matrix.inverse()
        sig_prime = (Winv + Sinv).inverse()
        mu_prime = (sig_prime @ (
            (Winv @ self.Xobs.T).T + Sinv @ self.prior.loc
        ).T).T
        npdfs = MultivariateNormal(
            self.prior.loc,
            self.W + self.prior.covariance_matrix,
        ).log_prob(self.Xobs).exp()
        omega_prime = self.woodbury_vector * npdfs

        weights = omega_prime / omega_prime.sum()
        W_prime = weights * MultivariateNormal(
            self.prior.loc,
            sig_prime,
        ).log_prob(mu_prime).exp()

        W_pos = W_prime[W_prime > 0].sum()
        W_neg = W_prime[W_prime < 0].sum().abs()
        N_pos = int(W_pos / (W_pos + W_neg) * self.n_gaussians)
        N_neg = self.n_gaussians - N_pos
        idx_pos = W_prime[W_prime > 0].argsort(descending=True)[:N_pos]
        idx_neg = W_prime[W_prime < 0].argsort()[:N_neg]
        weights_pos = weights[W_prime > 0][idx_pos]
        weights_neg = weights[W_prime < 0][idx_neg].abs()
        weights = torch.cat([weights_pos, weights_neg])
        mu_pos = mu_prime[W_prime > 0][idx_pos]
        mu_neg = mu_prime[W_prime < 0][idx_neg]
        mu_mean = torch.cat([mu_pos, mu_neg])

        idx_weights = weights > (self.threshold * weights.sum())
        weights = weights[idx_weights]
        mu_mean = mu_mean[idx_weights]
        weights = weights / weights.sum()
        return weights, mu_mean, sig_prime

    def joint_pdf_mean(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - first/first+second: torch.tensor, the values of probability density function of approximated GP mean functions
        """
        d_x = len(x)

        x_AA = (torch.tile(self.mu_mean, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
            self.d_mean * d_x, self.n_dims
        )
        Npdfs_AA = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims).to(self.device),
            self.sig_mean,
            x_AA,
        ).reshape(d_x, self.d_mean)

        f_AA = self.w_mean.unsqueeze(0) * Npdfs_AA
        pdf = f_AA.sum(axis=1)
        return pdf

    def sampling_mean(self, n):
        """
        Args:
            - n: int, number of samples to be sampled.

        Returns:
            - samples: torch.tensor, the samples from approximated GP mean functions
        """
        cnts = (n * self.w_mean).type(torch.int)
        samples = torch.cat([
            MultivariateNormal(
                self.mu_mean[i],
                self.sig_mean,
            ).sample(torch.Size([cnt])).to(self.device)
            for i, cnt in enumerate(cnts)
        ])
        return samples


class LogRootAcquisitionFunction(GaussianCalc):
    def __init__(self, prior, model, device, n_gaussians=100, threshold=1e-5):
        """
        Inherited the functions from GaussianCalc.

        Args:
        - prior: torch.distributions, prior distribution
        - device: torch.device, cpu or cuda
        """
        super().__init__(prior, device)
        self.n_gaussians = n_gaussians    # number of Gaussians for uncertainty sampling
        self.threshold = threshold        # threshold to cut off the small weights
        self.update(model)

    def update(self, model):
        """
        Args:
        - model: gpytorch.models, function of GP model, typically model = self.gp.model in _basq.py
        """
        self.parameters_extraction(model)
        self.w_A, self.w_B, self.w_C, self.mu_A, self.mu_B, self.mu_C, self.cov_A, self.cov_B, self.cov_C = self.sparseGMM_mean()
        self.d_A, self.d_B, self.d_C = len(self.mu_A), len(self.mu_B), len(self.mu_C)
        self.w_cov, self.mu_cov, self.sig_cov = self.sparseGMM()
        self.d_cov = len(self.mu_cov)

    def sparseGMM_mean(self):
        """
        Returns:
            - w_A, w_B, w_C: torch.tensor, the weights of first, second, and third terms of Gaussians
            - mu_A, mu_B, mu_C: torch.tensor, the mean vectors of first, second, and third terms of Gaussians
            - cov_A, cov_B, cov_C: torch.tensor, the covariance matrix of first, second, and third terms of Gaussians
        """
        X_ij_minus = (self.Xobs.unsqueeze(1) - self.Xobs.unsqueeze(0)).reshape(self.n_data**2, self.n_dims)
        Npdfs = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims).to(self.device),
            2 * self.W,
            X_ij_minus,
        ).reshape(self.n_data, self.n_data)

        w = self.v * self.woodbury_vector
        w_prime = 0.5 * (self.v**2) * self.woodbury_inv * Npdfs
        weights = torch.cat([w[w > 0], w_prime[w_prime < 0].abs(), torch.tensor(1).unsqueeze(0).to(self.device)])

        idx = weights.argsort(descending=True)[:self.n_gaussians]
        S = weights[idx].sum()
        idx_all = (weights[idx] / S > self.threshold)
        S = weights[idx[idx_all]].sum()

        mu_pi = self.prior.loc
        cov_pi = self.prior.covariance_matrix
        X_ij_plus = (self.Xobs.unsqueeze(1) + self.Xobs.unsqueeze(0)) / 2
        thresh1 = w[w > 0].size(0)
        thresh2 = thresh1 + w_prime[w_prime < 0].size(0)

        idx_update = idx[idx_all]
        condition1 = idx_update < thresh1
        condition2 = (idx_update > thresh1) * (idx_update < thresh2)
        condition3 = idx_update == thresh2
        if condition1.any():
            _idx = idx_update[condition1]
            w_A = weights[_idx] / S
            mu_A = self.Xobs[w > 0][_idx]
            cov_A = self.W
        else:
            w_A = torch.tensor([])
            mu_A = torch.tensor([])
            cov_A = torch.tensor([])

        if condition2.any():
            _idx = idx_update[condition2] - thresh1
            w_B = weights[idx_update[condition2]] / S
            mu_B = X_ij_plus[w_prime < 0][_idx]
            cov_B = self.W / 2
        else:
            w_B = torch.tensor([])
            mu_B = torch.tensor([])
            cov_B = torch.tensor([])

        if condition3.any():
            _idx = idx_update[condition3]
            w_C = weights[_idx] / S
            mu_C = mu_pi
            cov_C = cov_pi
        else:
            w_C = torch.tensor([])
            mu_C = torch.tensor([])
            cov_C = torch.tensor([])
        return w_A, w_B, w_C, mu_A, mu_B, mu_C, cov_A, cov_B, cov_C

    def joint_pdf_mean(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - pdf_sum: torch.tensor, the values of probability density function of approximated log (mu_g(x) pi(x))
        """
        d_x = len(x)
        pdf_sum = 0

        if not self.d_A == 0:
            x_A = (torch.tile(self.mu_A, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
                self.d_A * d_x, self.n_dims
            )
            Npdfs_A = self.utils.safe_mvn_prob(
                torch.zeros(self.n_dims).to(self.device),
                self.cov_A,
                x_A,
            ).reshape(d_x, self.d_A)

            f_A = self.w_A.unsqueeze(0) * Npdfs_A
            pdf_sum += f_A.sum(axis=1)

        if not self.d_B == 0:
            x_B = (torch.tile(self.mu_B, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
                self.d_B * d_x, self.n_dims
            )
            Npdfs_B = self.utils.safe_mvn_prob(
                torch.zeros(self.n_dims).to(self.device),
                self.cov_B,
                x_B,
            ).reshape(d_x, self.d_B)

            f_B = self.w_B.unsqueeze(0) * Npdfs_B
            pdf_sum += f_B.sum(axis=1)

        if not self.d_C == 0:
            Npdfs_C = self.utils.safe_mvn_prob(
                self.prior.loc,
                self.prior.covariance_matrix,
                x,
            )
            f_C = self.w_C * Npdfs_C
            pdf_sum += f_C

        return pdf_sum

    def sampling_mean(self, n):
        """
        Args:
            - n: int, number of samples to be sampled.

        Returns:
            - samples: torch.tensor, the samples from approximated GP mean functions
        """
        samples = torch.tensor([]).to(self.device)

        if not self.d_A == 0:
            cnts = (n * self.w_A).type(torch.int)
            samples_A = torch.cat([
                MultivariateNormal(
                    self.mu_A[i],
                    self.cov_A,
                ).sample(torch.Size([cnt])).to(self.device)
                for i, cnt in enumerate(cnts)
            ])
            samples = torch.cat([samples, samples_A])

        if not self.d_B == 0:
            cnts = (n * self.w_B).type(torch.int)
            samples_B = torch.cat([
                MultivariateNormal(
                    self.mu_B[i],
                    self.cov_B,
                ).sample(torch.Size([cnt])).to(self.device)
                for i, cnt in enumerate(cnts)
            ])
            samples = torch.cat([samples, samples_B])

        if not self.d_C == 0:
            cnt = (n * self.w_C).type(torch.int)
            samples_C = MultivariateNormal(
                self.mu_C,
                self.cov_C,
            ).sample(torch.Size([cnt])).to(self.device)
            samples = torch.cat([samples, samples_C])

        return samples

    def sparseGMM(self):
        """
        Returns:
            - w_A: torch.tensor, the weights of Gaussians
            - mu_A: torch.tensor, the mean vector of Gaussians
            - cov_A: torch.tensor, the covariance matrix of Gaussians
        """
        X_ij_plus = (self.Xobs.unsqueeze(1) + self.Xobs.unsqueeze(0)).reshape(self.n_data**2, self.n_dims)
        self.X_ij_plus_half = X_ij_plus / 2
        X_ij_minus = (self.Xobs.unsqueeze(1) - self.Xobs.unsqueeze(0)).reshape(self.n_data**2, self.n_dims)
        mu_pi = self.prior.loc
        cov_pi = self.prior.covariance_matrix
        sig_prime = (cov_pi.inverse() + 2 * self.W.inverse()).inverse()
        mu_prime = (sig_prime @ (cov_pi @ mu_pi + (self.W.inverse() @ X_ij_plus.T).T).T).T

        Npdfs_Xij_2W = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims).to(self.device),
            2 * self.W,
            X_ij_minus,
        ).reshape(self.n_data, self.n_data)

        Npdfs_Xij_W_prior = self.utils.safe_mvn_prob(
            mu_pi,
            self.W / 2 + cov_pi,
            self.X_ij_plus_half,
        ).reshape(self.n_data, self.n_data)

        mij = (self.v**2) * self.woodbury_vector.unsqueeze(1) @ self.woodbury_vector.unsqueeze(0) * Npdfs_Xij_2W * Npdfs_Xij_W_prior
        Wij = (self.v**2) * self.woodbury_inv * Npdfs_Xij_2W
        mij = mij.reshape(mij.numel())

        n_eff = int(1.1 * self.n_gaussians)
        idx = mij.argsort(descending=True)[:n_eff]

        self.Wij_flat = Wij.reshape(Wij.numel())
        Mij = torch.squeeze(mij[idx] * (self.calc_Taylor(mu_prime[idx]) * torch.exp(self.lengthscale) - 1))
        indice = idx[Mij.argsort(descending=True)][:self.n_gaussians]
        Mij = Mij[Mij.argsort(descending=True)][:self.n_gaussians]

        S_norm = Mij[Mij > 0].sum()
        if S_norm == 0:
            S_norm = 1
        w_A = Mij / S_norm
        
        idx_thresh = w_A > self.threshold
        if idx_thresh.sum() == 0:
            idx_thresh = torch.tensor([0])
            w_A = torch.tensor([1])
        else:
            w_A = w_A[idx_thresh]
            w_A /= w_A.sum()
        
        mu_A = mu_prime[indice][idx_thresh]
        cov_A = sig_prime
        
        return w_A, mu_A, cov_A

    def joint_pdf(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - first/first+second: torch.tensor, the values of probability density function of approximated A(x)
        """
        d_x = len(x)

        x_cov = (torch.tile(self.mu_cov, (d_x, 1, 1)) - x.unsqueeze(1)).reshape(
            self.d_cov * d_x, self.n_dims
        )
        Npdfs_cov = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims).to(self.device),
            self.sig_cov,
            x_cov,
        ).reshape(d_x, self.d_cov)

        f_cov = self.w_cov.unsqueeze(0) * Npdfs_cov
        pdf_sum = f_cov.sum(axis=1)
        return pdf_sum

    def sampling(self, n):
        """
        Args:
            - n: int, number of samples to be sampled.

        Returns:
            - samples: torch.tensor, the samples from approximated GP mean functions
        """
        samples = torch.tensor([])

        cnts = (n * self.w_cov).type(torch.int)
        samples = torch.cat([
            MultivariateNormal(
                self.mu_cov[i],
                self.sig_cov,
            ).sample(torch.Size([cnt])).to(self.device)
            for i, cnt in enumerate(cnts)
        ])
        return samples
