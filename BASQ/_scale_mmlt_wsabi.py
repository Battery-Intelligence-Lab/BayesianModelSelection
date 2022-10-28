import copy
import torch
from ._gp import update_gp
from ._utils import Utils
from ._gaussian_calc import GaussianCalc
from ._mmlt_wsabi import MmltWsabiGP


class ScaleMmltWsabiGP(MmltWsabiGP):
    def __init__(
        self,
        Xobs,
        Yobs,
        gp_kernel,
        device,
        label="wsabim",
        alpha_factor=1,
        lik=1e-10,
        training_iter=10000,
        thresh=0.01,
        lr=0.1,
        rng=10,
        train_lik=False,
        optimiser="L-BFGS-B",
    ):
        """
        Scaled-MMLT-WSABI BQ modelling
        Scaled-MMLT-WSABI is a scaled MMLT-WSABI BQ modelling.
        Scaling protects GP modelling from wide dynamic range of log-likelihood.
         ____________________________________________ _______________________________________ __________________
        |                  f space                   |               g space                 |     h space      |
        |____________________________________________|_______________________________________|__________________|
        |                     f                      |           g = sqrt(2(f - α))          |  h = log(g + 1)  |
        |               f = α + 1/2g^2               |              g = exp(h) - 1           |        h         |
        |          f = α + 1/2 exp(h)exp(h)          |              g = exp(h) - 1           |        h         |
        |____________________________________________|_______________________________________|__________________|
        |              f = GP(μ_f, σ_f)              |           g = GP(μ_g, σ_g)            | h = GP(μ_h, σ_h) |
        |          μ_f = α + 1/2(μ_g^2 + σ_g)        |        μ_g = exp(μ_h + 1/2 σ_h) - 1   |                  |
        | σ_f = 1/2σ_g(x,y)^2 + μ_g(x)σ_g(x,y)μ_g(y) | σ_g = μ_g(x)μ_g(y)(exp(σ_h(x,y)) - 1) |                  |
        |____________________________________________|_______________________________________|__________________|

        where β = max(Yobs_log), α = min(exp(Yobs_log - β)).
        - quadrature
            - log mean of marginal likelihood := log E[Z|f] + β
            - log variance of marginal likelihood := log Var[Z|f] + 2β

        WsabiMmltGP class summarises the functions of training, updating the warped GP model.
        This also provides the prediction and kernel of WSABI GP.
        The modelling of WSABI-L and WSABI-M can be easily switched by changing "label".
        The above table is the case of WSABI-M.

        Args:
           - Xobs: torch.tensor, X samples, X belongs to prior measure.
           - Yobs: torch.tensor, Y observations, Y = true_likelihood(X).
           - gp_kernel: gpytorch.kernels, GP kernel function
           - device: torch.device, device, cpu or cuda
           - label: string, the wsabi type, ["wsabil", "wsabim"]
           - lik: float, the initial value of GP likelihood noise variance
           - train_iter: int, the maximum iteration for GP hyperparameter training.
           - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
           - lr: float, the learning rate of Adam optimiser
           - rng: int, tne range coefficient of GP likelihood noise variance
           - train_like: bool, flag whether or not to update GP likelihood noise variance
           - optimiser: string, select the optimiser ["L-BFGS-B", "Adam"]
        """
        self.gp_kernel = gp_kernel
        self.device = device
        self.alpha_factor = 1
        self.alpha = alpha_factor
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik = train_lik
        self.optimiser = optimiser

        self.jitter = 0  # 1e-6
        self.Y_log = copy.deepcopy(Yobs)
        self.utils = Utils(device)

        self.model = update_gp(
            Xobs,
            self.process_y_warping_with_scaling(Yobs),
            gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
            optimiser=self.optimiser,
        )
        self.gauss = GaussianCalc(self.model, self.device)

    def process_y_warping_with_scaling(self, y_obs):
        """
        Args:
           - y_obs: torch.tensor, observations of true_loglikelihood

        Returns:
           - y_h: torch.tensor, warped observations in h space that contains no anomalies and the updated alpha hyperparameter.
        """

        y = self.utils.remove_anomalies(y_obs)
        self.beta = torch.max(y)
        y_f = torch.exp(y - self.beta)
        self.alpha = self.alpha_factor * torch.min(y_f)
        y_h = self.warp_from_f_to_h(y_f)
        return y_h

    def cat_observations_with_scaling(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, unwarped Y observations to be added to the existing data Yobs

        Returns:
           - Xall: torch.tensor, X samples that contains all samples
           - Yall: torch.tensor, warped Y observations that contains all observations
        """
        Xobs = self.model.train_inputs[0]
        Yobs_log = copy.deepcopy(self.Y_log)
        if len(self.model.train_targets.shape) == 0:
            Yobs_log = Yobs_log.unsqueeze(0)
        Xall = torch.cat([Xobs, X])
        Yall_log = torch.cat([Yobs_log, Y])
        self.Y_log = copy.deepcopy(Yall_log)
        Yall_h = self.process_y_warping_with_scaling(Yall_log)
        return Xall, Yall_h

    def update_mmlt_gp_with_scaling(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, unwarped Y observations to be added to the existing data Yobs
        """
        X_h, Y_h = self.cat_observations_with_scaling(X, Y)
        self.model = update_gp(
            X_h,
            Y_h,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
            optimiser=self.optimiser,
        )

    def retrain_gp_with_scaling(self):
        X_h = self.model.train_inputs[0]
        Y_h = self.process_y_warping_with_scaling(copy.deepcopy(self.Y_log))
        self.model = update_gp(
            X_h,
            Y_h,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
            optimiser=self.optimiser,
        )
