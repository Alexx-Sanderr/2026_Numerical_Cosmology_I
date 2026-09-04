import numpy as np


def extraction(config_dict):
    
    sne_path = config_dict["data_supernovae"]
    cov_path = config_dict["data_cov"]
    
    print(f"Loading data from: {sne_path}")
    print(f"Loading covariance from: {cov_path}")
          
    z, mu_obs, mu_err = np.loadtxt(sne_path, usecols=(1, 2, 3), unpack=True)

    covariance_data = np.loadtxt(cov_path)

    assert len(z) == 580, f"Expected 580 SNe, got {len(z)}"
    assert covariance_data.shape == (len(z), len(z)), "Covariance matrix shape incompatible with N_SNe"

    print("Data obtained: z, mu_obs, mu_err")
    print(f"SNe loaded: {len(z)}")
    print(f"Covariance shape: {covariance_data.shape}")
    
    return z, mu_obs, mu_err, covariance_data