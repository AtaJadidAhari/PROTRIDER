from .model import (_forward_outrider_in_batches, ModelInfo, MSEBCELoss,
                    NegativeBinomialLoss, OmicAutoencoder,
                    outrider_expected_counts, train, train_val)
from .model_helper import find_latent_dim, init_model

# Alias for backward compatibility
ProtriderAutoencoder = OmicAutoencoder
