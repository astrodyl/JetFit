import arviz as az


def flatten_chain(chain, n_dims: int):
    """  """
    return chain.reshape((-1, n_dims))


def create_data_dictionary(chain, parameters: list) -> dict:
    """ Creates a dictionary with key : value pairs of name : chains.
    Values have dimensions of (walkers, iterations).

    :param chain: emcee chain
    :param parameters: list of parameter names
    :return: key : value pairs of name : chains
    """
    return {name: chain[:, :, i] for i, name in enumerate(parameters)}


def create_inference_data(chain, parameters: list, **kwargs):
    """ Creates an inference data from an 'emcee' chain. If using PTSampler,
    pass only a chain from a single temperature.

    :param chain: emcee chain
    :param parameters: list of parameter names
    :param kwargs:
    :return: InferenceData object
    """
    return az.from_dict(posterior=create_data_dictionary(chain, parameters), **kwargs)
