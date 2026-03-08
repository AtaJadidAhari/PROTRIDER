from .model import train, train_val, MSEBCELoss, OmicAutoencoder, ModelInfo, NegativeBinomialLoss
from .model_helper import find_latent_dim, init_model

# Alias for backward compatibility
ProtriderAutoencoder = OmicAutoencoder #TODO
