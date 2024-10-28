import arviz as az
from matplotlib import pyplot as plt

from jetfit.analysis.utils import create_inference_data


"""
    Analysis.py
"""


class Analysis:
    def __init__(self, chain, parameters: list, **kwargs):
        self.chain = chain
        self.parameters = parameters
        self.inference_data = create_inference_data(chain, parameters, **kwargs)

        az.style.use("arviz-darkgrid")

    def summary(self, **kwargs):
        """ """
        return az.summary(self.inference_data, **kwargs)

    def conclusion(self, **kwargs):
        """ Call functions and make determination if results are good. """
        summary = az.summary(self.inference_data, **kwargs)

    def gelman_rubin(self):
        """ """
        return az.rhat(self.inference_data)

    def plot(self, **kwargs) -> None:
        """ Plots auto correlation, effective sample size, and trace. """
        self.plot_autocorrelation(**kwargs)
        self.plot_ess(**kwargs)
        self.plot_trace(**kwargs)

    def plot_autocorrelation(self, **kwargs) -> None:
        """ """
        az.plot_autocorr(self.inference_data, **kwargs)
        return plt.savefig(kwargs.get('output')) if kwargs.get('output') else None

    def plot_ess(self, **kwargs) -> None:
        """  """
        az.plot_ess(self.inference_data, **kwargs)
        return plt.savefig(kwargs.get('output')) if kwargs.get('output') else None

    def plot_trace(self, **kwargs) -> None:
        """  """
        az.plot_trace(self.inference_data, **kwargs)
        return plt.savefig(kwargs.get('output')) if kwargs.get('output') else None
