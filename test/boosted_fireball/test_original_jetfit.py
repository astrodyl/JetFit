import time
import unittest

import numpy as np

from jetfit.core.defns.evidence import Evidence
from jetfit.core.input import Observation
from jetfit.models.afterglow.boosted_fireball.boosted_fireball import BoostedFireball
from jetfit.models.afterglow.boosted_fireball.parameters.parameters import BFModelParams
from jetfit.models2.boosted import BoostedFireballModel


class MyTestCase(unittest.TestCase):
    def test_something(self):
        """"""
        evidence = Evidence.from_csv(r"C:\Projects\repos\JetFit\jetfit\resources\gws\170817\170817.csv")
        observation = Observation.from_csv(r"C:\Projects\repos\JetFit\jetfit\resources\gws\170817\170817_new.csv")

        # From old jetfit
        P = {'E': 0.15869069395227384, 'Eta0': 9.919507247518492, 'GammaB': 11.623593656572611, 'dL': 0.012188, 'epsb': 0.013323706571267526, 'epse': 0.04072783842837688, 'n': 0.0009871221028954489, 'p': 2.1333493591554804, 'theta_obs': 0.45459998935453005, 'xiN': 1.0, 'z': 0.00973}
        Spectral = ['5.51006881e-15 1.40362881e-15 1.60252607e-15 1.86203709e-14, 1.69002637e-14 8.50121155e-15 3.70080903e-15']

        original_flux = Spectral[0].replace(',', '')
        original_flux = original_flux.split(' ')
        original_flux = np.asarray([float(f) for f in original_flux])

        theta = {
            'explosion_energy': P['E'],
            'asymptotic_lorentz_factor': P['Eta0'],
            'circumburst_density': P['n'],
            'redshift': P['z'],
            'boost_lorentz_factor': P['GammaB'],
            'obs_angle': P['theta_obs'],
            'luminosity_distance': P['dL'],
            'electron_energy_index': P['p'],
            'accelerated_electron_fraction': P['xiN'],
            'electron_energy_fraction': P['epse'],
            'magnetic_energy_fraction': P['epsb'],
            'ebv_milky_way': 0,
            'ebv_source_frame': 0
        }

        params = BFModelParams(**theta)

        model = BoostedFireball()

        start1 = time.time()
        for _ in range(1_000):
            modeled = model.evaluate(evidence, params)
        stop1 = time.time()

        print(stop1 - start1, '\n')

        new_model = BoostedFireballModel(
            E=P['E'], n=P['n'], boost=P['GammaB'], eta=P['Eta0'], p=P['p'], zeta=P['xiN'],
            eps_e=P['epse'], eps_b=P['epsb'], z=P['z'], obs_angle=P['theta_obs'], dL=P['dL'],
            hydro_sim_table=model.scaler.hydro_sim_table
        )

        start2 = time.time()
        for _ in range(1_000):
            new_modeled = new_model.model(observation)
        stop2 = time.time()
        print(stop2 - start2, '\n')


        # chi_squared = -0.5 * maths.chi_squared(
        #     modeled,
        #     evidence.optimized_y,
        #     evidence.optimized_err,
        #     None
        # )
        #
        print()

if __name__ == '__main__':
    unittest.main()
