import numpy as np
from scipy import constants

A_TO_BOHR = np.divide(
    constants.physical_constants["Angstrom star"][0],
    constants.physical_constants["Bohr radius"][0]
    )

TAB2 = "      "