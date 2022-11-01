import copy
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
    sigma_omega = torch.tensor(6.07285737991333)
    ba = 2*torch.tensor(20.723264694213867)

    return w1, w2, t1, t2, Rt, r0, sigma_noise, ba, sigma_omega

def symmetric_KL(w1, w2, tau1, tau2, sigma_omega):
    Nis = 10000000

    if tau1 > tau2:
        temp = copy.deepcopy(tau1)
        tau1 = copy.deepcopy(tau2)
        tau2 = copy.deepcopy(temp)
        temp = copy.deepcopy(w1)
        w1 = copy.deepcopy(w2)
        w2 = copy.deepcopy(temp)

    samples = torch.from_numpy(stats.hypsecant.rvs(size=Nis))
    Nmid = int(Nis / 2)
    N1 = int(Nmid * w1)
    N2 = int(Nis - Nmid - N1)
    delta12 = sigma_omega * (w2*tau2 - w1*tau1)

    samples_N1 = samples[Nmid:Nmid+N1]/w1 - sigma_omega * tau1
    samples_N2 = samples[Nmid+N1:]/w2 - sigma_omega * tau2
    samples_Nmid = samples[:Nmid]*2 - 0.5 * delta12 - sigma_omega * w1 * tau1
    samples_g = torch.cat([samples_Nmid, samples_N1, samples_N2])

    # importance sampling
    P1 = w1/torch.pi/torch.cosh(w1*(samples_g + sigma_omega * tau1))
    P2 = w2/torch.pi/torch.cosh(w2*(samples_g + sigma_omega * tau2))
    g = 1/4*(
        w1/torch.pi/torch.cosh(w1*(samples_g + sigma_omega * tau1))
        + w2/torch.pi/torch.cosh(w2*(samples_g + sigma_omega * tau2))
    ) + 1/4/torch.pi/torch.cosh(
        0.5*(samples_g + sigma_omega * w1 * tau1 + 0.5*delta12)
    )
    KL12 = ((P1/P2).log()*P1/g).mean()
    KL21 = ((P2/P1).log()*P2/g).mean()
    Dkl = KL12 + KL21
    return Dkl

def signal_to_noise(w1, w2, t1, t2, Rt, r0, sigma_noise, ba, sigma_omega):
    W12 = w1 * w2
    Wsum = w1**2 + w2**2
    Delta_mu_12 = sigma_omega * torch.abs(t1 - t2)
    
    EZ = Rt * torch.pi * (1 - r0) / ba
    A = Wsum + 2 * W12 * Delta_mu_12 / torch.sinh(Delta_mu_12)
    EZ2 = (Rt**2) * ((1-r0)**2) / ba * A
    SNR = torch.log((EZ2 - EZ**2) / sigma_noise)
    return SNR

def calc_uncertainties(params_true):
    w1, w2, t1, t2, Rt, r0, sigma_noise, ba, sigma_omega = calc_params(params_true)
    KL = symmetric_KL(w1, w2, t1, t2, sigma_omega)
    SNR = signal_to_noise(w1, w2, t1, t2, Rt, r0, sigma_noise, ba, sigma_omega)
    rho = SNR * KL
    return torch.tensor([KL, SNR, rho]).float()