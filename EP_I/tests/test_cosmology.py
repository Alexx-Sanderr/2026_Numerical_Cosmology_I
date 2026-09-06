import numpy as np
import pytest
import yaml

from numcosmo_I.cosmology import FLRW
from pathlib import Path


@pytest.fixture
def yaml_config():
    config_dir = Path( __file__ ).resolve().parent.parent / "config"

    yaml_files = list( config_dir.glob( "*.yaml" ) ) + list( config_dir.glob( "*.yaml" ) )

    if not yaml_files:
        pytest.fail(f"No .yaml configuration file was found in: {config_dir}")
        
    yaml_path = yaml_files[0]
    
    with open(yaml_path, "r", encoding="utf-8") as file:
        dict_yaml = yaml.safe_load( file )
        
    return dict_yaml

@pytest.fixture
def cosmo_model(yaml_config):
    
    cosmology_config = yaml_config[ "cosmology" ]
    
    return FLRW(
        h = cosmology_config[ "h" ],
        Omega_m = cosmology_config[ "Omega_m" ],
        Omega_lambda = cosmology_config[ "Omega_lambda" ],
        Omega_r = cosmology_config[ "Omega_r" ],
        w0_fld = cosmology_config[ "w0_fld" ],
        wa_fld = cosmology_config[ "wa_fld" ],
    )


def test_expansion_at_present(cosmo_model):
    """
        Ensures E(z=0) equal to 1.0
    """
    
    val = cosmo_model.E(0.0)
    
    assert val == pytest.approx(1.0, abs=1e-10), f" Value expected = 1.0 ; Returned value: {val}"

def test_etherington_duality(cosmo_model):
    """
        Etherington duality test D_L(z) = ( 1 + z )^2 + D_A ( z ) 
    """
    
    z  = 0.5
    dL = cosmo_model.luminosity_distance( z )
    dA = cosmo_model.angular_diameter_distance( z )
    
    assert dL == pytest.approx( ( 1.0 + z ) ** 2 * dA, rel = 1e-6)

def test_flat_universe_transverse_distance():
    """
        Ensures D_M(z) == chi(z) when Omega_k = 0
    """
    
    flat_model = FLRW(
        h = 0.7,
        Omega_m = 0.3,
        Omega_lambda = 0.7,
        Omega_r = 0.0,
        w0_fld = -1.0,
        wa_fld = 0.0,
    )
    
    z   = 0.8
    chi = flat_model.comoving_distance( z )
    DM  = flat_model.transverse_comoving_distance( z )

    assert abs( flat_model.O_curvature ) < 1e-6
    assert DM == pytest.approx( chi, abs = 1e-10 )

def test_vectorization_support(cosmo_model):
    """
         NumPy vectorization input and same-size arrays output
    """
    
    z_array = np.array( [ 0.1, 0.5, 1.0 ] )
    E_array = cosmo_model.E( z_array )

    assert isinstance( E_array, np.ndarray )
    assert E_array.shape == z_array.shape

def test_missing_yaml_key_raises_error(yaml_config):
    """
        YAML nonexistent key raises KeyError
    """
    
    with pytest.raises(KeyError):
        _ = yaml_config[ "cosmology" ][ "nonexistent_key" ]


def test_invalid_parameter_type_raises_error():
    """
        Invalid type raises ValueError
    """
    
    with pytest.raises(ValueError):
        FLRW(
            h = "invalid_string",
            Omega_m = 0.3,
            Omega_lambda = 0.7,
            Omega_r = 0.0,
            w0_fld = -1.0,
            wa_fld = 0.0,
        )