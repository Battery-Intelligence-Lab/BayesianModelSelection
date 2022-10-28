import torch
import scipy.stats as stats

def calc_params(params_true):
    Rt = torch.exp(params_true[0])
    _r1 = torch.exp(-torch.exp(params_true[1]))
    t1 = params_true[2]
    _r2 = torch.exp(-torch.exp(params_true[3]))
    t2 = params_true[4]
    sigma_noise = torch.exp(-torch.exp(params_true[5]))
    r0 = 1 - _r1 - _r2

    w1 = _r1 / (1 - r0)
    w2 = _r2 / (1 - r0)

    W12 = w1 * w2
    gamma_12 = w1/w2
    Delta_mu_12 = torch.abs(t1 - t2)
    
    return W12, gamma_12, Delta_mu_12

def mutual_information(W12, gamma_12, Delta_mu_12):
    samples = torch.from_numpy(stats.hypsecant.rvs(size=10000000))
    Z12 = torch.mean(1/torch.cosh(samples + Delta_mu_12)*W12/torch.pi)
    csch = 1/torch.sinh(Delta_mu_12)
    zeta = torch.mean(
        1/torch.pi / torch.cosh(samples) \
        * W12/torch.pi / torch.cosh(samples + Delta_mu_12) \
        * torch.log(W12/torch.pi / torch.cosh(samples + Delta_mu_12))
    )
    I12 = -torch.pi - 2*W12*Delta_mu_12/(torch.pi*Z12)*csch * torch.log(Z12) + zeta/Z12
    return I12

def signal_to_noise(params_true):
    Rt = torch.exp(params_true[0])
    _r1 = torch.exp(-torch.exp(params_true[1]))
    _r2 = torch.exp(-torch.exp(params_true[3]))
    sigma_noise = torch.exp(-torch.exp(params_true[5]))
    r0 = 1 - _r1 - _r2
    SNR = torch.log(Rt * torch.pi * (1 - r0) / (2*sigma_noise))
    return SNR
    
def calc_uncertainties(params_true):
    W12, gamma_12, Delta_mu_12 = calc_params(params_true)
    MI = mutual_information(W12, gamma_12, Delta_mu_12)
    SNR = signal_to_noise(params_true)
    #effD = EffectiveDataSize(params_true, true_loglikelihood, model)
    #effd = effD.robust_effD()
    rho = SNR / MI  # / effd
    return torch.tensor([MI, SNR, rho]).float()