import torch
from torch.special import gammaln, digamma

from .estimate_theta_robust_moments import estimate_theta_robust_moments

__all__ = ['NegativeBinomialDistribution']

class NegativeBinomialDistribution:
    def __init__(self, x_true):
        # distribution.mom should return numpy array or tensor with initial dispersions
        theta = estimate_theta_robust_moments(counts=x_true.T, theta_min=1e-2, theta_max=1e3)
        self.dispersions = torch.tensor(theta, dtype=torch.float64)
        print(f"INIT dispersions = {self.dispersions}")

    # TODO: make fit paralel, i.e. not n_parallel=1, but vectorize everything
    # TODO: maybe using torch.vmap ? https://docs.pytorch.org/docs/stable/generated/torch.vmap.html
    def fit(self, x_true, x_pred, n_parallel=1):
        """
        x_true, x_pred: torch.Tensor, shape (genes, samples)
        """
        genes = x_true.shape[0]
        fitted_disps = torch.empty(genes, dtype=torch.float64)

        for gene_index in range(genes):
            fitted_disps[gene_index] = self._fit_dispersion_single(x_true[gene_index], x_pred[gene_index], gene_index=gene_index)

        self.dispersions = fitted_disps

        print(f"after fitting dispersions = {self.dispersions}")
        return self.dispersions

    def _fit_dispersion_single(self, x_true_gene, x_pred_gene, gene_index, max_iter=100, lr=0.1, min_disp=1e-2, max_disp=1e6):
        # TODO: double check the clamping
        # Initialize directly from current dispersion value
        disp = self.dispersions[gene_index].clone().detach().to(torch.float64).requires_grad_(True) 
        optimizer = torch.optim.LBFGS([disp], max_iter=max_iter, line_search_fn='strong_wolfe')
    
        x_true_gene = x_true_gene.to(torch.float64)
        x_pred_gene = x_pred_gene.to(torch.float64)
    
        def closure():
            optimizer.zero_grad()
            # Keep it in valid range during optimization
            disp_clamped = torch.clamp(disp, min_disp, max_disp)
            loss = self.loss(dispersion=disp_clamped, x_true=x_true_gene, x_pred=x_pred_gene)
            loss.backward()
            return loss
    
        optimizer.step(closure)
    
        final_disp = torch.clamp(disp.detach(), min_disp, max_disp)
        return final_disp

    def loss(self, dispersion, x_true, x_pred):
        # Negative log-likelihood for NB for one gene (dispersion scalar)
        # x_true and x_pred: tensors, shape (samples,)
        r = dispersion
        mu = x_pred

        # NB log PMF terms:
        # gammaln(x + r) - gammaln(r) - gammaln(x+1) + r * ln(r/(r+mu)) + x * ln(mu/(r+mu))
        term1 = gammaln(x_true + r) - gammaln(r) - gammaln(x_true + 1)
        term2 = r * torch.log(r / (r + mu))
        term3 = x_true * torch.log(mu / (r + mu))
        log_prob = term1 + term2 + term3

        return -log_prob.sum()

    def get_dispersions(self):
        return None if self.dispersions is None else self.dispersions.detach().cpu().numpy()
