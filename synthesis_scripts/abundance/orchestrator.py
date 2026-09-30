"""
This script:
    - reads the configuration file for a given star
    - reads the file containing the lines from which to derive the abundance
    - for each line (for a given element, for a given model atmosphere, for given parameters):
        (maybe delegated to another script)
        - reads the line list
        - reads the file containing the NLTE departure coefficients
        - reads the model atmosphere
        - computes multiple synthetic spectra for different abundances
        - compares the synthetic spectra to the observed spectrum
        - computes the abundance from the closest synthetic spectrum
    - stores the derived abundances in a file

"""

### READING THE FILES AND CONFIGURATION
import os
import sys
import yaml
from pathlib import Path
import numpy as np

selected_star = "HD196944"

# Paths
stars_config_file = Path("config/stars.yaml")
kept_lines_file = Path(f"data/selected_lines/{selected_star}/fe_kept_lines.csv")

# Check if the files exist
if not stars_config_file.exists():
    print(f"Error: {stars_config_file} does not exist.")
    sys.exit(1)

if not kept_lines_file.exists():
    print(f"Error: {kept_lines_file} does not exist.")
    sys.exit(1)

# Read the configuration and kept lines
with open(stars_config_file, "r") as f:
    config = yaml.safe_load(f)

kept_lines = np.loadtxt(kept_lines_file, delimiter=",", dtype=str, skiprows=1)

# Teff = config[selected_star]["Teff"]
# logg = config[selected_star]["logg"]
# feoh = config[selected_star]["feoh"]
# nlte_elt = config[selected_star]["nlte_elt"]
# nlte_abundance = config[selected_star]["nlte_abundance"]
# sph = config[selected_star]["sph"]


#### FUNCTION TO GET THE ATMOSPHERE FILE

ATM_PATH = Path("data/original/atm")
INTERP_ATM_PATH = Path(f"data/interpolated/{selected_star}")

def get_atmosphere_file(Teff, logg, feoh, nlte_elt, nlte_abundance, sph, m):
    """
    Constructs the path to the model atmosphere file based on given parameters and checks if it exists. 
    If the file does not exist, it prints a message and interpolates the missing data using nlte_interp.py.
    """
    # Construct the filename based on the parameters
    filename = "s" if sph else "p"
    filename += f"{Teff:.0f}_g+{logg:.1f}_m{m:.1f}_t02_st_z{feoh:.2f}_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00"
    type_dir = "MARCS_st_sph_t02_mod" if sph else "MARCS_st_ppl_t01_mod"
    atmosphere_file = ATM_PATH / type_dir / filename

    # Check if the atmosphere file exists
    if not atmosphere_file.with_name(atmosphere_file.name + '.mod').exists():
        print(f"Error: Atmosphere file {atmosphere_file.with_name(atmosphere_file.name + '.mod')} does not exist.")
    else:
        print(f"Atmosphere file {atmosphere_file.with_name(atmosphere_file.name + '.mod')} exists.")
        quit()

    if not atmosphere_file.with_name(atmosphere_file.name + '.int').exists():
        print(f"Interpolated atmosphere file {atmosphere_file.with_name(atmosphere_file.name + '.int')} does not exist. Interpolating...")
    else:
        print(f"Interpolated atmosphere file {atmosphere_file.with_name(atmosphere_file.name + '.int')} exists.")
        quit()


get_atmosphere_file(2500, 1.0, -3.00, "fe", -3.00, True, 1.0)