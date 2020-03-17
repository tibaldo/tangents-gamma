import numpy as np

##spiral arm model from Reid+ 19

Narm = 17

arms = ['/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/3kF_lbvRBD.2019', # 3kpc
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/3kN_lbvRBD.2019', # 3kpc
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/N1F_lbvRBD.2019', # Norma/outer
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/N1N_lbvRBD.2019', # Norma/outer
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/N4N_lbvRBD.2019', # Norma/outer
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/N4F_lbvRBD.2019', # Norma/outer
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/Out_lbvRBD.2019', # Norma/outer
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/ScN_lbvRBD.2019', # Scutum/Cen/OSC
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/ScF_lbvRBD.2019', # Scutum/Cen/OSC
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/CtN_lbvRBD.2019', # Scutum/Cen/OSC
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/CtF_lbvRBD.2019', # Scutum/Cen/OSC
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/OSC_lbvRBD.2019', # Scutum/Cen/OSC
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/SgF_lbvRBD.2019',  # Sgr/Car
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/SgN_lbvRBD.2019',  # Sgr/Car
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/CrN_lbvRBD.2019',  # Sgr/Car
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/Loc_lbvRBD.2019',  # local
        '/Users/ltibaldo/Fermi/ISM/BeSSeL/v2.4.1_bundle/Per_lbvRBD.2019',  # Perseus
        ]

def arm_polar(s):
    dat = np.genfromtxt(arms[s], comments='!')
    rr = dat[:,3]
    tt = dat[:,4]
    # convert azimuth to radians
    # and change sign to match my convention
    tt = -np.deg2rad(tt)
    return rr, tt
