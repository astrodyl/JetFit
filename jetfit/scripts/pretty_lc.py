import json

from dust_extinction.parameter_averages import CCM89

from jetfit.core import utils
from jetfit.core.input import Observation
from jetfit.models.fireball import StratifiedFireballModel, FireballModel
from jetfit.scripts.plot.light_curve import LightCurvePlot


spread = {
    '210905A': {
        'K_offset': 10,
        'H_offset': 8,
        'J_offset': 6,
        'Ic_offset': 4,
        'i_offset': 2
    },
    '220101A': {
        'K_offset': 20,
        'H_offset': 16,
        'J_offset': 12,
        'F125W_offset': 12,
        'F775W_offset': 2,
        'I_offset': 8,
        'R_offset': 6,

        'z_offset': 4,
        'i_offset': 2
    },
    '131030A': {
        'H_offset': 8,
        'J_offset': 6,
        'R_offset': 4,
        'B_offset': 2,

        'z_offset': 2,
        'i_offset': 1.5,
    }
}


def main(model, params, observation, event):
    """"""
    # Plot light curve
    lc = LightCurvePlot(
        model=model,
        params=params,
        observation=observation,
    )
    lc.plot(
        ext_model=CCM89(Rv=3.1),
        show=True,
        # spread=spread.get(event)
    )


if __name__ == "__main__":

    event_name = '080413B'
    params_path = rf"C:\Projects\repos\JetFit\jetfit\results\{event_name}\best_fit.json"

    # Read in params
    with open(params_path, "r") as f:
        params = json.load(f)

    main(**{
        'model':
            StratifiedFireballModel,

        'params':
            params,

        'observation':
            Observation.from_csv(
                utils.get_input_csv_path('custom', event_name)
            ),

        'event':
            event_name

    })
