import abc
from typing import Optional
import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
import torch
import torch.optim as optim
import torch.nn as nn

from .estimate_theta_robust_moments import estimate_theta_robust_moments, estimate_rho_robust_moments

__all__ = ['Dispersion', 'NegativeBinomialDistribution']

class Dispersion():
    def __new__(cls, analysis, *args, **kwargs):
        if cls is Dispersion:   # Only auto-redirect when calling the base class
            if analysis == "fraser":
                return FraserDispersion(*args, **kwargs)
            elif analysis == "outrider":
                return OutriderDispersion(*args, **kwargs)
            else:
                raise ValueError(f"Unknown analysis type: {analysis}")
        return super().__new__(cls)
    def __init__(self, analysis, distribution):
        self.analysis = analysis  # needed?
        self.distribution = distribution


class OutriderDispersion():
    def __init__(self, distribution: Optional[str] = None):
        if distribution is None:
            distribution = NegativeBinomialDistribution()
        #super().__init__(analysis='outrider', distribution=distribution)
        self.distribution = distribution
        self.mu_scale = None
        self.theta = None

    def get_parameters(self):
        mu_scale = None if self.mu_scale is None else self.mu_scale.detach().cpu().numpy()
        theta = None if self.theta is None else self.theta.detach().cpu().numpy()
        return mu_scale, theta
    
    def set_dispersion(self, theta):
        self.theta = theta

    def clip_theta(self, lower=0.01, upper=1000):
        self.theta = torch.clip(self.theta, lower, upper)

    def fit(self, x_true, x_pred, max_iter=100, lower_bound=0.01, device=None):
        """
        x_true, x_pred: torch.Tensor, shape (genes, samples)
        """

        x_true = x_true.to(dtype=torch.float64, device=device)
        x_pred = x_pred.to(dtype=torch.float64, device=device)
        
        mu_scale_init, theta_init = self.distribution.init_fit(x_true, x_pred)
        
        # Reparameterization using exp, since LBFGS does not consider bounds:
        # where theta = exp(p_theta) + lower_bound
        # This ensures parameters >= lower_bound
        # Use min=1e-8 to avoid log(0)
        p_theta_init = torch.log(torch.clamp(theta_init - lower_bound, min=1e-8))
        p_mu_scale_init = torch.log(torch.clamp(mu_scale_init - lower_bound, min=1e-8))

        p_theta = nn.Parameter(p_theta_init.to(device))
        p_mu_scale = nn.Parameter(p_mu_scale_init.to(device))

        optimizer = optim.LBFGS(
            [p_theta, p_mu_scale],
            max_iter=max_iter,
            history_size=5,
            tolerance_change=2.2e-9,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            
            # Transform parameters (shape [genes])
            theta = torch.exp(p_theta) + lower_bound
            mu_scale = torch.exp(p_mu_scale) + lower_bound
            
            # Unsqueeze to [genes, 1] for broadcasting against data [genes, samples]
            theta = theta.unsqueeze(1)
            mu_scale = mu_scale.unsqueeze(1)

            # Calculate loss and backpropagate
            mu = x_pred * mu_scale
            loss = self.distribution.loss(x_true, theta, mu)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.theta = (torch.exp(p_theta) + lower_bound).detach().cpu()
        self.mu_scale = (torch.exp(p_mu_scale) + lower_bound).detach().cpu()

class FraserDispersion(): 
    def __init__(self, distribution: Optional[str] = None):
        if distribution is None:
            distribution = BetaBinomialDistribution()
        self.distribution = distribution
        self.mu = None
        self.rho = None

    def get_parameters(self):
        mu = None if self.mu is None else self.mu.detach().cpu().numpy()
        rho = None if self.rho is None else self.rho.detach().cpu().numpy()
        return mu, rho
    
    def set_dispersion(self, rho):
        self.rho = rho

    def clip_rho(self, lower=0.01, upper=1000): #ASK
        self.rho = torch.clip(self.rho, lower, upper)

    def clip_mu(self, lower=0.01, upper=0.99): #ASK
        self.mu = torch.clip(self.mu, lower, upper)


    def fit(self, K, N, x_pred, rho_min=1e-5, rho_max=1-1e-5, lambda_penalty=1e-4, max_iter=100,
            logit_bound=30.0, tol=1e-7):
        mu = torch.sigmoid(x_pred).T.to(torch.float64)   # (junctions, samples)
        K = K.to(torch.float64)                          # (junctions, samples)
        N = N.to(torch.float64)                          # (junctions, samples)
        _, rho_init = self.distribution.init_fit(K, N)   # (junctions,)

        logit_min, logit_max = -logit_bound, logit_bound

        logit_rho = torch.logit(rho_init.to(torch.float64).clamp(rho_min, rho_max)).detach()

        def eval_loss(lr):
            rho = torch.sigmoid(lr).unsqueeze(-1)
            return self.distribution.loss_penalized(K, N, mu, rho, lambda_penalty)

        damping = torch.full_like(logit_rho, 1e-3)
        loss_curr = eval_loss(logit_rho).detach()

        for _ in range(max_iter):
            with torch.enable_grad():
                logit_rho.requires_grad_(True)
                rho = torch.sigmoid(logit_rho).unsqueeze(-1)  # (junctions, 1)
                loss = self.distribution.loss_penalized(K, N, mu, rho, lambda_penalty)  # (junctions,)
                grad, = torch.autograd.grad(loss.sum(), logit_rho, create_graph=True)
                hess, = torch.autograd.grad(grad.sum(), logit_rho)

            # damped Hessian (never near-zero) instead of a hard floor
            hess_damped = hess.abs() + damping
            step = torch.clamp(grad / hess_damped, -5.0, 5.0)
            candidate = (logit_rho.detach() - step).clamp(logit_min, logit_max)

            cand_loss = eval_loss(candidate).detach()
            improved = cand_loss < loss_curr

            # only accept steps that actually reduce the loss; else keep old value
            logit_rho = torch.where(improved, candidate, logit_rho.detach()).detach()
            loss_curr = torch.where(improved, cand_loss, loss_curr)
            damping = torch.where(improved, damping * 0.7, damping * 2.0).clamp(1e-6, 1e6)

            if step.abs().max() < tol:
                break

        with torch.no_grad():
            rho_final = torch.sigmoid(logit_rho)
            rho_final = torch.where(torch.isfinite(rho_final), rho_final, rho_init.to(torch.float64))

        self.rho = rho_final.detach()
        self.mu = mu

class Distribution(): # Do we need this base class?
    #@abc.abstractmethod
    def init_train(self):
        pass

    #@abc.abstractmethod
    def init_fit(self):
        pass

    #@abc.abstractmethod
    def loss(self):
        pass

class NegativeBinomialDistribution(Distribution):
    def init_train(self, x_true, theta_min=0.01, theta_max=1000.0):
        """Initialize theta and mu for training: theta robust moments"""
        theta = estimate_theta_robust_moments(x_true=x_true.T, theta_min=theta_min, theta_max=theta_max)
        mu_scale = None
        return mu_scale, theta

    def init_fit(self, x_true, size_factors, epsilon=1e-8, theta_min=0.01, theta_max=1000.0, mu_min=0.01):
        """
        Initialize theta and mu for fitting: theta is dispersion, mu_scale the mean
        x_true, size_factors: torch.Tensor, shape (genes, samples)
        """
        x_true = x_true.to(torch.float64)
        size_factors = size_factors.to(torch.float64)
        normalized = x_true / (size_factors + epsilon)
        
        # Calculate mean and var per gene (dim=1)
        mu_scale = normalized.mean(dim=1)
        var = normalized.var(dim=1)

        theta = mu_scale**2 / (var - mu_scale + epsilon)

        # Set fallback for genes where var <= mu_scale or theta is invalid
        fallback_mask = (var <= mu_scale) | (theta <= 0) | torch.isnan(theta)
        theta[fallback_mask] = 1.0
        
        theta = torch.clamp(theta, theta_min, theta_max)
        mu_scale = torch.clamp(mu_scale, min=mu_min)
        return mu_scale, theta

    def loss(self, x_true, theta, mu):
        term_lgamma = torch.lgamma(x_true + theta) - torch.lgamma(theta) - torch.lgamma(x_true + 1)
        term_t_log_t = torch.xlogy(theta, theta)
        term_x_log_m = torch.xlogy(x_true, mu)
        term_tx_log_tm = torch.xlogy(x_true + theta, theta + mu)
        
        log_prob = term_lgamma + term_t_log_t + term_x_log_m - term_tx_log_tm
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-1e20)
        loss = -torch.sum(log_prob)
        return loss

class BetaBinomialDistribution(Distribution):
    def init_train(self, K, N):
        """Initialize mu and rho for training."""
        rho = estimate_rho_robust_moments(K, N ,rho_min=1e-5, rho_max=1-1e-5) 
        mu = None
        return mu, rho

    def init_fit(self, K, N, epsilon=1e-8, mu_min=1e-4,rho_min=1e-5, rho_max=1-1e-5):
        """Initialize mu and rho for fitting."""
        K = K.to(torch.float64)
        N = N.to(torch.float64)

        p = K / (N + epsilon)

        mu = p.mean(dim=1)
        var = p.var(dim=1)

        # method-of-moments rho
        denom = mu * (1 - mu)
        rho = (var - denom / (N.mean(dim=1) + epsilon)) / (denom + epsilon)

        # fallback if invalid
        fallback = (rho <= 0) | torch.isnan(rho) | torch.isinf(rho)
        rho[fallback] = 0.1

        mu = torch.clamp(mu, mu_min, 1 - mu_min)
        rho = torch.clamp(rho, rho_min, rho_max)

        return mu, rho

    def loss(self, K, N, mu, rho, eps = 0.0):
        """Negative log-likelihood using mu/rho parameterization."""
        K = K.to(torch.float64)
        N = N.to(torch.float64)

        # convert to alpha/beta
        conc = (1 - rho) / rho
        alpha = mu * conc
        beta = (1 - mu) * conc

        log_prob = (
            torch.lgamma(N + 1 + 2 * eps)
            - torch.lgamma(K + 1 + eps)
            - torch.lgamma(N - K + 1 + eps)
            + torch.lgamma(K + alpha + eps)
            + torch.lgamma(N - K + beta + eps)
            - torch.lgamma(N + alpha + beta + 2 * eps)
            + torch.lgamma(alpha + beta)
            - torch.lgamma(alpha)
            - torch.lgamma(beta)
        )

        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=-1e20)
        return -torch.mean(log_prob, dim = -1) #Sum to mean?

    def loss_penalized(self, K, N, mu, rho, lambda_penalty = 1e-4):
        """Negative log-likelihood (per junction) with an L2 penalty on
        logit(rho), also per junction -- matches the per-row shape
        returned by loss(), so junctions stay independent (no cross-junction
        sum) when used elementwise/broadcast over the junction dimension."""
        nll = self.loss(K, N, mu, rho)              # (junctions,)
        logit_rho = torch.logit(rho).squeeze(-1)     # (junctions,), match nll's shape
        penalty = lambda_penalty * logit_rho ** 2
        return nll + penalty