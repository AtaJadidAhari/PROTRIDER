import copy
from typing import Optional

import torch
from torch.special import gammaln
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from ..dispersions import Dispersion, NegativeBinomialDistribution

from protrider.datasets import ProtriderSubset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,  # show DEBUG and above
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@dataclass
class ModelInfo:
    """Stores model information."""
    q: np.array
    learning_rate: np.array
    n_epochs: np.array
    test_loss: np.array
    train_losses: np.array
    df0: np.array = None  # Degrees of freedom for the t-distribution, if applicable
    df_folds: Optional[pd.DataFrame] = None  # DataFrame with fold assignments (for CV runs)
    
    def save(self, out_dir: str) -> None:
        """
        Save model information to CSV files.
        
        Args:
            out_dir: Output directory path
        """
        import pandas as pd
        import dataclasses
        from pathlib import Path
        
        logger.info('=== Saving model info ===')
        
        out_dir = Path(out_dir)
        model_info_dict = dataclasses.asdict(self)
        
        # Remove df_folds from model_info_dict to handle separately
        df_folds = model_info_dict.pop("df_folds", None)
        
        if self.q.ndim == 0:
            # make all variables of model_info arrays
            model_info_dict = {k: np.array([v]) for k, v in model_info_dict.items()}

        folds = np.arange(len(model_info_dict['q']))
        
        # Remove train_losses from dict (handled separately for non-CV runs)
        train_losses = model_info_dict.pop("train_losses")
        
        # Save training losses separately if not doing CV
        if df_folds is None and len(train_losses) > 0:
            train_losses_df = pd.DataFrame({
                'epoch': range(1, len(train_losses[0]) + 1),
                'train_loss': train_losses[0],
            })
            out_p = out_dir / 'train_losses.csv'
            train_losses_df.to_csv(out_p, header=True, index=False)
            logger.info(f"Saved training losses to {out_p}")

        # Save additional info
        out_p = out_dir / 'additional_info.csv'
        df_info = pd.DataFrame(model_info_dict, index=pd.Index(folds, name='fold'))
        df_info.to_csv(out_p, header=True, index=True)
        logger.info(f"Saved additional info to {out_p}")

        # Save folds if provided
        if df_folds is not None:
            out_p = out_dir / 'folds.csv'
            df_folds.to_csv(out_p, header=True, index=True)
            logger.info(f"Saved folds to {out_p}")
    
    def plot_training_loss(self, out_dir: str = None, **kwargs):
        """
        Plot training loss history.
        
        Args:
            out_dir: Optional output directory for saving plots. If None, plot is returned but not saved.
            **kwargs: Additional arguments passed to the plotting function (plot_title, fontsize)
            
        Returns:
            plotnine plot object
            
        Example:
            >>> plot = model_info.plot_training_loss()  # Interactive use
            >>> plot.draw()
            >>> model_info.plot_training_loss(out_dir='output/')  # Save plot
        """
        from protrider import plots
        # Build train_losses DataFrame
        train_losses_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
        })
        return plots.plot_training_loss(
            output_dir=out_dir,
            train_losses=train_losses_df,
            **kwargs
        )

class ConditionalEnDecoder(nn.Module):

    def __init__(self, in_dim, out_dim, is_encoder, h_dim=None, n_layers=1,
                 prot_means=None):
        super().__init__()
        self.prot_means = prot_means

        self.is_encoder = is_encoder
        self.n_layers = n_layers

        last_layer = None
        if n_layers == 1:
            # if the model is a decoder, then we want to have trainable bias
            last_layer = nn.Linear(in_dim,
                                   out_dim,
                                   bias=not is_encoder or prot_means is None)
            self.model = last_layer

        elif n_layers > 1:
            modules = []
            modules.append(nn.Linear(in_dim, h_dim, bias=False))
            modules.append(nn.ReLU())
            for _ in range(1, n_layers - 1):
                modules.append(nn.Linear(h_dim, h_dim, bias=False))
                modules.append(nn.ReLU())
            # if the model is a decoder, then we want to have trainable bias
            last_layer = nn.Linear(h_dim, out_dim, bias=not is_encoder or prot_means is None)
            modules.append(last_layer)
            self.model = nn.Sequential(*modules)

        # if the model is a decoder, then the bias should be initialized to the protein means
        if not is_encoder and prot_means is not None:
            last_layer.bias.data.copy_(prot_means).squeeze(0)

    def forward(self, x, cond=None):
        if self.is_encoder and (self.prot_means is not None):
            x = x - self.prot_means
        if cond is not None:
            x = torch.cat([x, cond], -1)
        return self.model(x)


### input: s x (g + cov)
### encoder: (g+cov) x h
### latent: s x h
### decoder:(h+cov) x g
### output: s x g

class ProtriderAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim, n_layers=1, n_cov=0, h_dim=None,
                 prot_means=None, presence_absence=False, model_type="protrider"):
        super().__init__()
        self.model_type = model_type
        self.n_layers = n_layers
        self.presence_absence = presence_absence
        self.encoder = ConditionalEnDecoder(in_dim=in_dim + n_cov,
                                            out_dim=latent_dim, h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=True, prot_means=prot_means)

        self.decoder = ConditionalEnDecoder(in_dim=latent_dim + n_cov,
                                            out_dim=in_dim,
                                            h_dim=h_dim, n_layers=n_layers,
                                            is_encoder=False, prot_means=prot_means)
        
        if self.model_type == "outrider":
            self.distribution = NegativeBinomialDistribution()
            self.dispersion = Dispersion(self.distribution)
            

    def forward(self, x, mask, cond=None):
        if self.presence_absence:
            presence = (~mask).double()
            x = torch.stack([x, presence])
            cond = torch.stack([cond, cond])

        z = self.encoder(x, cond=cond)
        out = self.decoder(z, cond=cond)

        if self.model_type == "outrider":
            out = torch.clip(out, -700, 700)

        return out

    def initialize_wPCA(self, Vt_q, prot_means, n_cov=0):
        if self.n_layers > 1:
            logger.warning('Initialization only possible for n_layers=1. Going back to random init...')
            return

        device = self.encoder.model.weight.device
        Vt_q = torch.from_numpy(Vt_q).to(device) # (q, n_prots)

        ## ENCODER weights: (q, n_prots + n_cov), bias: (q)
        cov_enc_init = self.encoder.model.weight.data[:, 0:n_cov]
        self.encoder.model.weight.data.copy_(
            torch.cat([Vt_q.to(device),
                       cov_enc_init.to(device)], axis=1)
        )

        if self.model_type == "outrider":
            self.encoder.model.bias.data.copy_(0)
        elif self.model_type == "protrider":
            self.encoder.model.bias.data.copy_(-(Vt_q @ torch.from_numpy(prot_means).to(device).T).flatten())

        ## DECODER weights: (n_prots, q + n_cov), bias: (n_prot)
        self.decoder.model.bias.data.copy_(torch.from_numpy(prot_means).squeeze(0))
        cov_dec_init = self.decoder.model.weight.data[:, 0:n_cov]
        self.decoder.model.weight.data.copy_(
            torch.cat([Vt_q.T.to(device),
                       cov_dec_init.to(device)], axis=1)
        )      

    def update_dispersion(self, x_true, x_pred):
        self.dispersion.update(x_true, x_pred)

    def fit_dispersion(self, x_true, x_pred):
        self.dispersion.fit(x_true, x_pred)

    def get_dispersion_parameters(self):
        """Get dispersion parameters (mu_scale, theta)"""
        return self.dispersion.get_parameters()


def mse_masked(x_hat, x, mask):
    reconstruction_loss = nn.MSELoss(reduction="none")
    loss = reconstruction_loss(x_hat, x)
    masked_loss = torch.where(mask, torch.nan, loss)
    reconstruction_loss_val = masked_loss.nanmean()
    return reconstruction_loss_val


class MSEBCELoss(nn.Module):
    def __init__(self, presence_absence=False, lambda_bce=1.):
        super().__init__()
        self.presence_absence = presence_absence
        self.lambda_bce = lambda_bce

    def forward(self, x_hat, x, mask, detached=False):
        if self.presence_absence:
            presence = (~mask).double()
            # n = x_hat.shape[1] // 2
            presence_hat = x_hat[1]  # Predicted presence (0–1)
            x_hat = x_hat[0]  # Predicted intensities

        reconstruction_loss = mse_masked(x_hat, x, mask)
        if detached:
            reconstruction_loss = reconstruction_loss.detach().cpu().numpy()

        if self.presence_absence:
            bce_loss = F.binary_cross_entropy(torch.sigmoid(presence_hat), presence)
            if detached:
                bce_loss = bce_loss.detach().cpu().numpy()
            loss = reconstruction_loss + self.lambda_bce * bce_loss
        else:
            bce_loss = None
            loss = reconstruction_loss

        return loss, reconstruction_loss, bce_loss



def train_val(train_subset: ProtriderSubset, val_subset: ProtriderSubset, model, criterion, n_epochs=100, learning_rate=1e-3, val_every_nepochs=1,
              batch_size=None, patience=100, min_delta=0.001):
    # start data;pader
    if batch_size is None:
        batch_size = train_subset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_val_loss = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping_counter = 0
    early_stopping_epoch = 0

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs)):
        train_loss, train_reconstruction_loss, train_bce_loss = _train_iteration(data_loader, model, criterion, optimizer)

        if epoch % val_every_nepochs == 0:
            train_losses.append(train_loss)
            x_hat_val = model(val_subset.X, val_subset.torch_mask, cond=val_subset.covariates)
            val_loss, val_reconstruction_loss, val_bce_loss = criterion(x_hat_val, val_subset.X, val_subset.torch_mask)

            val_losses.append(val_loss.detach().cpu().numpy())
            logger.debug('[%d] train loss: %.6f' % (epoch + 1, train_loss))
            logger.debug('[%d] validation loss: %.6f' % (epoch + 1, val_loss))

            if min_val_loss - val_loss > min_delta:
                min_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                early_stopping_counter = 0
                early_stopping_epoch = epoch + 1
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    logger.info(f"\tEarly stopping at epoch {epoch + 1}")
                    break

    logger.info('\tRestoring model weights from epoch %s', early_stopping_epoch)
    model.load_state_dict(best_model_wts)
    # make losses a 2d array
    return np.array(train_losses), np.array(val_losses)


def train(dataset, model, criterion, n_epochs=100, learning_rate=1e-3, batch_size=None):
    # start data;pader
    if batch_size is None:
        batch_size = dataset.X.shape[0]
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)

    best_model_wts = copy.deepcopy(model.state_dict())  # placeholder
    best_loss = 10**9
    train_losses = []
    for epoch in tqdm(range(n_epochs)):
        running_loss, running_reconstruction_loss, running_bce_loss = _train_iteration(data_loader, model, criterion, optimizer)
        logger.debug('[%d] loss: %.6f, reconstruction loss: %.6f, bce loss: %.6f' % (epoch + 1, running_loss,
                                                                          running_reconstruction_loss, running_bce_loss))
        scheduler.step()
        if running_loss < best_loss:
            best_loss = running_loss
            best_model_wts = copy.deepcopy(model.state_dict())  # save weights
        train_losses.append(running_loss)
    
    model.load_state_dict(best_model_wts)
    
    return running_loss, running_reconstruction_loss, running_bce_loss, train_losses


def _train_iteration(data_loader, model, criterion, optimizer):
    running_loss = 0.0
    running_reconstruction_loss = 0.0
    running_bce_loss = 0.0

    n_batches = 0
    for batch_idx, data in enumerate(data_loader):
        if model.model_type == "protrider":
            x, mask, cov, prot_means = data
        elif model.model_type == "outrider":
            x, mask, cov, prot_means, raw_x, size_factors = data

        # Restore grads and compute model out
        optimizer.zero_grad()
        x_hat = model(x, mask, cond=cov)

        # Calculate loss
        if model.model_type == "outrider":
            _, theta = model.get_dispersion_parameters()

            x_pred = torch.exp(x_hat) * size_factors
            loss, reconstruction_loss, bce_loss = criterion((theta, x_pred), raw_x, mask)
        elif model.model_type == "protrider":
            loss, reconstruction_loss, bce_loss = criterion(x_hat, x, mask) 

        # Adjust learning weights
        loss.backward()
        optimizer.step()

        # Update dispersions in OUTRIDER model
        if model.model_type == "outrider":
            with torch.no_grad():
                _, theta = model.get_dispersion_parameters()
                x_pred = torch.exp(x_hat) * size_factors
                model.fit_dispersion(raw_x.T, x_pred.T)
                model.dispersion.clip_theta()
                _, theta = model.get_dispersion_parameters()

        # Gather data and report
        running_loss += loss.item()
        running_reconstruction_loss += reconstruction_loss.item()
        running_bce_loss += bce_loss.item() if bce_loss is not None else 0
        n_batches += 1

    return running_loss / n_batches, running_reconstruction_loss / n_batches, running_bce_loss / n_batches

class NegativeBinomialLoss(nn.Module):
    def __init__(self, presence_absence=False, lambda_bce=1.0, eps=1e-8):
        super().__init__()
        self.presence_absence = presence_absence
        self.lambda_bce = lambda_bce
        self.eps = eps

    def forward(self, predictions, x_true, mask=None, detached=False):
        """
        predictions: tuple of (theta, x_pred) where:
            - theta: dispersion parameters (genes,) 
            - x_pred: predicted counts (batch_size x genes)
        x_true: observed counts (batch_size x genes)
        mask: optional mask
        """

        if isinstance(predictions, tuple):
            theta, x_pred = predictions
        else:
            raise ValueError("Theta must be provided for NB loss")
        
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, dtype=torch.float64, device=x_true.device)
        
        # Ensure theta has the right shape (1, genes) for broadcasting
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        
        # Compute NB negative log-likelihood per gene
        eps = 1e-10
        r = torch.clamp(theta, min=eps)
        mu = torch.clamp(x_pred, min=eps)
        
        term1 = gammaln(x_true + r) - gammaln(r) - gammaln(x_true + 1)
        term2 = r * torch.log(r / (r + mu))
        term3 = x_true * torch.log(mu / (r + mu))
        log_prob = term1 + term2 + term3
        
        if mask is not None:
            log_prob = torch.where(mask, torch.tensor(0.0, device=log_prob.device), log_prob)
            nll = -log_prob.sum() / (~mask).sum()
        else:
            nll = -log_prob.mean()
        
        # Handle presence/absence if needed
        if self.presence_absence:
            presence = (~mask).double()
            presence_hat = x_pred[1]
            x_pred = x_pred[0]
            bce_loss = F.binary_cross_entropy(torch.sigmoid(presence_hat), presence)
            loss = nll + self.lambda_bce * bce_loss
        else:
            bce_loss = None
            loss = nll

        if detached:
            return (loss.detach().cpu().numpy(),
                    nll.detach().cpu().numpy(),
                    bce_loss.detach().cpu().numpy() if bce_loss is not None else None)
            
        return loss, nll, bce_loss
