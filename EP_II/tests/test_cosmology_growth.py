import numpy as np
import pytest

from numcosmo_II.cosmology import FLRW

@pytest.fixture
def eds_model():

    return FLRW( )