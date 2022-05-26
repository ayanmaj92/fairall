"""
Authors: Ayan Majumdar, Miriam Rateike
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

from .cvae import CVAE
from .sscvae import SSCVAE


def create_model(model_name, prob_model, conditional, cfg, phase):
    """
    Helper function to create the model object
    @param model_name: name of the type of model
    @param prob_model: probabilistic model object
    @param conditional: do we model conditional to sensitive S?
    @param cfg: the config dict
    @param phase: phase 1 or 2
    @return: return corresponding model object
    """
    if model_name == 'cvae':
        return CVAE(prob_model, cfg["model"]["params"]["latent_size"], cfg["model"]["params"]["h_dim_list_enc"],
                    cfg["model"]["params"]["h_dim_list_dec"], cfg["model"]["params"]["act_name"], conditional,
                    cfg["model"]["params"]["beta"], cfg["seed"], phase)
    elif model_name == 'sscvae':
        return SSCVAE(prob_model,
                      cfg["model"]["params"]["latent_size"],
                      cfg["model"]["params"]["h_dim_list_clf"],
                      cfg["model"]["params"]["h_dim_list_enc"],
                      cfg["model"]["params"]["h_dim_list_dec"],
                      cfg["model"]["params"]["act_name"],
                      cfg["model"]["params"]["drop_rate_clf"],
                      conditional,
                      cfg["model"]["params"]["alpha"],
                      cfg["model"]["params"]["costs"],
                      cfg["seed"],
                      cfg["model"]["params"]["beta"],
                      cfg["model"]["params"]["lambda"],
                      cfg["model"]["params"]["loss_function"],
                      cfg["model"]["params"]["model_type"],
                      phase)

    raise AssertionError(f'Model not found: {model_name}')


__all__ = ['create_model']
