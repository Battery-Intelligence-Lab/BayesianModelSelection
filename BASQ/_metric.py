import torch


class KLdivergence:
    def __init__(self, prior, test_data, Z_true, device, true_function):
        """
        Args:
            - prior: torch.distributions, prior distribution
            - test_data: torch.tensor, samples for evaluation, sampled from prior
            - Z_true: float, the true evidence.
            - device: torch.device, cpu or cuda
            - true_function: function of y = function(x), true likelihood funciton
        """
        self.prior = prior
        self.test_data = test_data
        self.Z_true = Z_true
        self.device = device
        self.true_function = true_function

    def __call__(self, basq_model):
        """
        Args:
            - basq_mode: gpytorch.models, the trained BQ model
            - EZy: float, the estimated evidence

        Returns:
            - KL: torch.tensor, the Kullback-Leibler divergence between true posterior and estimated posterior
        """
        KL = torch.zeros(len(self.test_data)).to(self.device)
        q = torch.squeeze(
            self.prior.log_prob(self.test_data).exp() * self.true_function(self.test_data).exp() / self.Z_true
        )
        p = basq_model.posterior.joint_posterior(self.test_data)
        KL[p > 0] = p[p > 0] * torch.log(p[p > 0] / q[p > 0])
        _KL = KL[KL.isinf().logical_not()]
        return torch.abs(_KL.sum() * len(self.test_data) / len(_KL))
