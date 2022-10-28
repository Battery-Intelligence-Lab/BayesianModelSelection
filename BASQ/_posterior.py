import torch


class Posterior:
    def __init__(self, bq_model, prior, gp, predict_mean, kq, sampler, sampler_type):
        """
        Args:
            - bq_model: string, ["mmlt", "wsabi", "vbq"]
            - prior: torch.distributions, prior distribution.
            - gp: class, Gaussian process module
            - predict_mean: function of mean = function(x), the function that returns the predictive mean at given x
            - kq: class, Kernel Quadrature module
            - sampler: class, sampler module
            - sampler_type: string, type of sampler
        """
        self.bq_model = bq_model
        self.prior = prior
        self.gp = gp
        self.predict_mean = predict_mean
        self.kq = kq
        self.sampler = sampler
        self.sampler_type = sampler_type

    def check_evidence(self):
        if not hasattr(self, "EZy"):
            if self.bq_model == "mmlt":
                if not hasattr(self.kq, "logEZy"):
                    logEZy, _ = self.kq.quadrature()
                else:
                    logEZy = self.kq.logEZy
                self.EZy = (logEZy - self.gp.beta).exp()
            else:
                if not hasattr(self.kq, "EZy"):
                    self.EZy, _ = self.kq.quadrature()
                else:
                    self.EZy = self.kq.EZy

    def joint_posterior(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - torch.tensor, the posterior of given x
        """
        self.check_evidence()
        return self.predict_mean(x) * self.prior.log_prob(x).exp() / self.EZy

    def sample(self, n):
        """
        Args:
            - n: int, number of samples to be generated

        Returns:
            - samples: torch.tensor, the samples drawn from posterior
        """
        if self.bq_model == "mmlt" and self.sampler_type == "uncertainty":
            self.sampler.update(self.gp.model)
            n_super = int(self.sampler.ratio_super * n)
            supersample = self.sampler.sampling(n_super)

            mu_log = self.predict_mean(supersample).detach().abs().log()
            prior_log = self.prior.log_prob(supersample)
            sampler_log = torch.nan_to_num(self.sampler.joint_pdf(supersample)).log()
            weights = torch.exp(
                mu_log + prior_log - sampler_log
            )
            samples = self.sampler.SIR(supersample, weights, n)
            return samples
        else:
            raise Exception("this feature has not been implemented yet.")

    def MAP_estimation(self, n):
        """
        Args:
            - n: int, number of seeds

        Returns:
            - X_map: torch.tensor, maximum a posteori sample
        """
        seeds = self.sample(n)
        ypred = self.joint_posterior(seeds)
        idx_max = ypred.argmax()
        MAP = ypred[idx_max].item()
        X_map = seeds[idx_max]
        print("PDF of posterior at MAP: " + str(MAP))
        return X_map
