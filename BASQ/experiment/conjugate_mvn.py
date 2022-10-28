import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class ConjugateMVN:
    def __init__(self, num_data, num_dim, amp, device):
        self.num_data = num_data
        self.num_dim = num_dim
        self.amp = amp
        self.device = device
        self.X_data, self.y_data = [], []
        self.random_hyperparameters()
        self.set_posterior()
        self.set_true_evidence()

    def random_hyperparameters(self):
        self.mu_true = self.mu_generator()
        self.cov_true = self.cov_generator()
        self.mu_prior = self.mu_generator()
        self.cov_prior = 2 * self.cov_generator()
        self.mvn_true = MultivariateNormal(self.mu_true, self.cov_true)
        self.mvn_prior = MultivariateNormal(self.mu_prior, self.cov_prior)

    def mu_generator(self):
        return self.amp * torch.rand(self.num_dim).to(self.device)

    def cov_generator(self):
        return self.amp * torch.diag(torch.rand(self.num_dim) + 1).to(self.device)

    def sampling_from_true_likelihood(self, n):
        return self.mvn_true.sample(torch.Size([n]))

    def true_likelihood(self, X):
        return self.mvn_true.log_prob(X).exp()

    def true_loglikelihood(self, X):
        return self.mvn_true.log_prob(X)

    def create_observed_data(self):
        self.X_data = self.sampling_from_true_likelihood(self.num_data)
        self.y_data = self.true_likelihood(self.X_data)

    def set_posterior(self):
        # posterior
        self.create_observed_data()
        self.N = self.X_data.size(0)
        self.cov_posterior = (self.N * self.cov_true.inverse() + self.cov_prior.inverse()).inverse()
        self.mu_posterior = self.cov_posterior @ (
            self.cov_true.inverse() @ self.X_data.sum(axis=0) + \
            self.cov_prior.inverse() @ self.mu_prior
        )
        self.mvn_posterior = MultivariateNormal(self.mu_posterior, self.cov_posterior)

    def set_true_evidence(self):
        X_test = self.mvn_prior.sample(torch.Size([10000]))
        y_true = self.posterior(X_test)
        y_true_unnormalised = self.prior(X_test) * self.__call__(X_test).exp()
        Cs = y_true_unnormalised / y_true
        self.true_log_evidence = (Cs[Cs.isnan().logical_not()].mean()).log()
        print("true log evidence is "+str(self.true_log_evidence.item()))

    def prior(self, X):
        return self.mvn_prior.log_prob(X).exp()

    def posterior(self, X):
        return self.mvn_posterior.log_prob(X).exp()

    def __call__(self, X):
        # log true likelihood
        d_x = len(X)
        X = (torch.tile(self.X_data, (d_x, 1, 1)) - X.unsqueeze(1)).reshape(self.num_data * d_x, self.num_dim).to(self.device)
        Npdfs = MultivariateNormal(
            torch.zeros(self.num_dim).to(self.device),
            self.cov_true,
        ).log_prob(X).exp().reshape(d_x, self.num_data)

        return Npdfs.prod(axis=1).log()
