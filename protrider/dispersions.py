import torch
from torch.special import gammaln, digamma
import pandas as pd

from .estimate_theta_robust_moments import estimate_theta_robust_moments

__all__ = ['Dispersion', 'NegativeBinomialDistribution']

class Dispersion:
    def __init__(self, distribution):
        self.distribution = distribution
        self.mu_scale = None
        self.theta = None

    def get_parameters(self):
        mu_scale = None if self.mu_scale is None else self.mu_scale.detach().cpu().numpy()
        theta = None if self.theta is None else self.theta.detach().cpu().numpy()
        return mu_scale, theta
    
    def update(self, x_true, x_pred):
        print("not yet implemented")

    # TODO: make fit paralel, i.e. not n_parallel=1, but vectorize everything
    # TODO: maybe using torch.vmap ? https://docs.pytorch.org/docs/stable/generated/torch.vmap.html
    def fit(self, x_true, x_pred, n_parallel=1):
        """
        x_true, x_pred: torch.Tensor, shape (genes, samples)
        """
        genes = x_true.shape[0]
        theta_fitted = torch.empty(genes, dtype=torch.float64)
        mu_scale_fitted = torch.empty(genes, dtype=torch.float64)

        for gene_index in range(genes):
            mu_scale_fitted[gene_index], theta_fitted[gene_index] = self._fit_gene(x_true=x_true[gene_index], x_pred=x_pred[gene_index])

        self.theta = theta_fitted
        self.mu_scale = mu_scale_fitted

        print("storing values after fitting")
        out_theta_path = "./intermediate_results/theta.csv"
        out_mu_scale_path = "./intermediate_results/mu_scale.csv"
        pd.DataFrame(self.theta.detach().cpu().numpy()).to_csv(out_theta_path, header=False, index=False)
        pd.DataFrame(self.mu_scale.detach().cpu().numpy()).to_csv(out_mu_scale_path, header=False, index=False)

    def _fit_gene(self, x_true, x_pred, max_iter=100, lower_bound=1e-2):
        mu_scale_init, theta_init = self.distribution.init_fitting(x_true, x_pred)
        mu_scale = torch.clamp(mu_scale_init, min=lower_bound)
        theta = torch.clamp(theta_init, min=lower_bound)
        
        mu_scale.requires_grad = True
        theta.requires_grad = True
        optimizer = torch.optim.LBFGS([theta, mu_scale], max_iter=max_iter, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()

            theta_clamped = torch.clamp(theta, min=lower_bound)
            scaled_predictions = x_pred * mu_scale
            scaled_predictions = torch.clamp(scaled_predictions, min=lower_bound)

            loss = self.distribution.loss(theta_clamped, x_true, scaled_predictions)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        mu_scale_final = mu_scale.detach()
        theta_final = theta.detach()
        return mu_scale_final, theta_final

class NegativeBinomialDistribution:
    def init_training(self, x_true, x_pred):
        """Initialize theta and mu for training: theta robust moments, mu_scale as the x_pred"""
        theta = estimate_theta_robust_moments(x_true=x_true.T, theta_min=1e-2, theta_max=1e3)
        mu_scale = None
        return mu_scale, theta

    def init_fitting(self, x_true, size_factors):
        """Initialize theta and mu for fitting: theta is dispersion, mu_scale the mean"""
        normalized = x_true / size_factors
        mu_scale = normalized.mean()
        var = normalized.var()

        if var > mu_scale:
            theta = mu_scale**2 / (var - mu_scale)
        else:
            theta = torch.tensor(1.0, dtype=torch.float64)

        return mu_scale, theta
    
    def loss(self, dispersion, x_true, x_pred):
        # Negative log-likelihood for NB per gene (dispersion scalar)
        # Ensure numerical stability with epsilon value
        eps = 1e-10
        r = torch.clamp(dispersion, min=eps)
        mu = torch.clamp(x_pred, min=eps)
        
        # NB log PMF terms
        # gammaln(x + r) - gammaln(r) - gammaln(x+1) + r * ln(r/(r+mu)) + x * ln(mu/(r+mu))
        term1 = gammaln(x_true + r) - gammaln(r) - gammaln(x_true + 1)
        term2 = r * torch.log(r / (r + mu))
        term3 = x_true * torch.log(mu / (r + mu))
        log_prob = term1 + term2 + term3
        
        return -log_prob.sum()
