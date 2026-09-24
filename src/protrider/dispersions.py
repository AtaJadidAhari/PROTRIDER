import abc
from typing import Optional
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
        self.theta = None
        self.mean_scale = None
        self._counts_cache = None

    def _count_constants(self, counts, lower_bound, upper_bound, batched=False):
        """Reuse count-only calculations until the tensor or its contents change."""
        key = (counts._version, counts.dtype, counts.device, lower_bound, upper_bound, batched)
        cached = self._counts_cache
        if cached is None or cached[0] is not counts or cached[1] != key:
            with torch.no_grad():
                # The robust estimate sorts the entire count matrix. Keep that
                # initialization off the GPU when the fit is memory bounded.
                init_counts = counts.cpu() if batched else counts
                theta_init = estimate_theta_robust_moments(
                    init_counts, lower_bound, upper_bound
                ).to(counts.device)
                count_lgamma = None if batched else torch.lgamma(counts + 1)
            self._counts_cache = (counts, key, theta_init, count_lgamma)
        return self._counts_cache[2:]

    def get_parameters(self):
        theta = None if self.theta is None else self.theta.detach().cpu().numpy()
        mean_scale = self.mean_scale
        if mean_scale is None and self.theta is not None:
            # A trained autoencoder supplies gene baselines through its decoder
            # bias, so its effective post-fit scale is one.
            mean_scale = torch.ones_like(self.theta)
        mean_scale = None if mean_scale is None else mean_scale.detach().cpu().numpy()
        return mean_scale, theta
    
    def set_dispersion(self, theta, mean_scale=None):
        self.theta = theta
        self.mean_scale = mean_scale

    def clip_theta(self, lower=0.01, upper=1000):
        self.theta = torch.clip(self.theta, lower, upper)

    def fit(self, x_true, x_pred, max_iter=100, lower_bound=0.01,
            upper_bound=1000.0, device=None, fit_mean_scale=False, batch_size=None):
        """
        x_true, x_pred: torch.Tensor, shape (samples, genes). ``x_pred`` is
        the full expected-count matrix, not a normalized mean.

        ``fit_mean_scale`` is reserved for PCA/no-training correction, where
        OUTRIDER fits a per-gene mean multiplier together with theta. A trained
        autoencoder uses its decoder bias instead.
        """

        if x_true.shape != x_pred.shape:
            raise ValueError("OUTRIDER theta fitting requires matching samples x genes matrices.")
        device = device or x_true.device
        dtype = x_pred.dtype
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        batched = batch_size is not None
        # Chunk reduction changes summation order. Accumulate the optimizer's
        # small parameter vectors in double precision so LBFGS line search is
        # stable even when the input and stored dispersion use float32.
        fit_dtype = torch.float64 if batched else dtype
        if not batched:
            x_true = x_true.to(dtype=dtype, device=device)
            x_pred = x_pred.to(dtype=dtype, device=device)
        theta_init, count_lgamma = self._count_constants(
            x_true, lower_bound, upper_bound, batched=batched
        )
        theta_init = theta_init.to(dtype=fit_dtype, device=device)
        # Expected counts remain fixed during theta-only optimization. The PCA
        # mean-scale fit must recompute this term as its mean changes.
        count_log_mu = None if fit_mean_scale or batched else torch.xlogy(x_true, x_pred)
        p_theta = nn.Parameter(torch.log(torch.clamp(theta_init - lower_bound, min=1e-8)))
        parameters = [p_theta]

        p_mean_scale = None
        if fit_mean_scale:
            if batched:
                with torch.no_grad():
                    scale_sum = torch.zeros_like(theta_init)
                    for start in range(0, len(x_true), batch_size):
                        stop = min(start + batch_size, len(x_true))
                        counts = x_true[start:stop].to(dtype=fit_dtype, device=device)
                        expected = x_pred[start:stop].to(dtype=fit_dtype, device=device)
                        scale_sum += (counts / torch.clamp(
                            expected, min=torch.finfo(fit_dtype).tiny
                        )).sum(dim=0)
                    mean_scale_init = (scale_sum / len(x_true)).clamp(min=lower_bound)
            else:
                mean_scale_init = torch.mean(
                    x_true / torch.clamp(x_pred, min=torch.finfo(dtype).tiny), dim=0
                ).clamp(min=lower_bound)
            p_mean_scale = nn.Parameter(
                torch.log(torch.clamp(mean_scale_init - lower_bound, min=1e-8))
            )
            parameters.append(p_mean_scale)

        optimizer = optim.LBFGS(
            parameters,
            max_iter=max_iter,
            history_size=5,
            tolerance_change=2.2e-9,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()

            if batched:
                total_loss = torch.zeros((), dtype=fit_dtype, device=device)
                for start in range(0, len(x_true), batch_size):
                    stop = min(start + batch_size, len(x_true))
                    counts = x_true[start:stop].to(dtype=fit_dtype, device=device)
                    expected = x_pred[start:stop].to(dtype=fit_dtype, device=device)
                    theta = torch.clamp(
                        torch.exp(p_theta) + lower_bound, max=upper_bound
                    ).unsqueeze(0)
                    if p_mean_scale is not None:
                        expected = expected * (torch.exp(p_mean_scale) + lower_bound).unsqueeze(0)
                    loss = self.distribution.loss(counts, theta, expected)
                    loss.backward()
                    total_loss += loss.detach()
                return total_loss
            else:
                theta = torch.clamp(torch.exp(p_theta) + lower_bound, max=upper_bound).unsqueeze(0)
                expected = x_pred
                if p_mean_scale is not None:
                    expected = expected * (torch.exp(p_mean_scale) + lower_bound).unsqueeze(0)
                if type(self.distribution) is NegativeBinomialDistribution:
                    loss = self.distribution.loss(
                        x_true, theta, expected,
                        count_lgamma=count_lgamma, count_log_mu=count_log_mu,
                    )
                else:
                    loss = self.distribution.loss(x_true, theta, expected)
                loss.backward()
                return loss

        optimizer.step(closure)

        self.theta = torch.clamp(
            torch.exp(p_theta) + lower_bound, max=upper_bound
        ).detach().to(dtype=dtype)
        self.mean_scale = (
            (torch.exp(p_mean_scale) + lower_bound).detach().to(dtype=dtype)
            if p_mean_scale is not None else None
        )

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
        mu = torch.sigmoid(x_pred).T.to(torch.float32)   # (junctions, samples)
        K = K.to(torch.float32)                          # (junctions, samples)
        N = N.to(torch.float32)                          # (junctions, samples)
        _, rho_init = self.distribution.init_fit(K, N)   # (junctions,)

        logit_min, logit_max = -logit_bound, logit_bound

        logit_rho = torch.logit(rho_init.to(torch.float32).clamp(rho_min, rho_max)).detach()

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
            rho_final = torch.where(torch.isfinite(rho_final), rho_final, rho_init.to(torch.float32))

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
        theta = estimate_theta_robust_moments(x_true=x_true, theta_min=theta_min, theta_max=theta_max)
        mu_scale = None
        return mu_scale, theta

    def init_fit(self, x_true, size_factors, epsilon=1e-8, theta_min=0.01, theta_max=1000.0, mu_min=0.01):
        """
        Initialize theta and mu for fitting: theta is dispersion, mu_scale the mean
        x_true, size_factors: torch.Tensor, shape (genes, samples)
        """
        dtype = size_factors.dtype
        x_true = x_true.to(dtype=dtype)
        size_factors = size_factors.to(dtype=dtype)
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

    def loss(self, x_true, theta, mu, *, count_lgamma=None, count_log_mu=None):
        if count_lgamma is None:
            count_lgamma = torch.lgamma(x_true + 1)
        term_lgamma = torch.lgamma(x_true + theta) - torch.lgamma(theta) - count_lgamma
        term_t_log_t = torch.xlogy(theta, theta)
        term_x_log_m = torch.xlogy(x_true, mu) if count_log_mu is None else count_log_mu
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
        K = K.to(torch.float32)
        N = N.to(torch.float32)

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
        K = K.to(torch.float32)
        N = N.to(torch.float32)

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
