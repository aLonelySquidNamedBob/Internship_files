#!/usr/bin/env python3

from pathlib import Path
import shutil
import subprocess
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths
home = Path("~/NICO").expanduser()
working_dir = home / "test" / "test1"       # test1 directory

# Program paths
turbospectrum = Path("~/NICO/Turbospectrum_NLTE_20.1").expanduser()
babsma  = turbospectrum / "exec-gf/babsma_lu"
bsyn    = turbospectrum / "exec-gf/bsyn_lu"
faltbon = turbospectrum / "Utilities/faltbon"
log     = working_dir / "log.txt"

# Config files
config_file = Path("config/stars.yaml")
nlte_config_file = Path("config/nlte.yaml")
line_list_file = Path("config/line_lists.yaml")

with config_file.open() as f:
    config: dict[str, dict[str, str]] = yaml.safe_load(f)

with nlte_config_file.open() as f:
    nlte_config: dict[str, dict[str, str]] = yaml.safe_load(f)

with line_list_file.open() as f:
    line_list_config: dict[str, dict[str, str]] = yaml.safe_load(f)

# stars = ["HD115444", "HD196944", "TIC396792499"]
stars = list(config.keys())
selected_star = stars[1]  # Change this to select a different star

line = 5194.9414
wmin = line - 10
wmax = line + 10
dw = 0.01  # wavelength step


# Departure coefficients
elt         = config[selected_star]["synthesis"]["NLTE_element"]
Z           = nlte_config["element"][elt]["atomic_number"]
model_atom  = nlte_config["element"][elt]["model_atom"]
atom_path   = Path(f"{nlte_config['element'][elt]['model_atom_path']}")
dc_path     = home / f"{config['stars'][selected_star]['synthesis']['departure_coefficients_path']}"
dc          = dc_path / config[selected_star]["synthesis"]["departure_coefficients"]


# ['F']: LTE, ['T']: NLTE
# ['F', 'T'] or ['T', 'F']: LTE and NLTE
nltes = ["T"]

# To change
sspath = home / "runs" / selected_star / "abundance_tests" / f"Fe_{line}"  # synthetic spectra path
copath = home / "runs" / selected_star / "abundance_tests" / f"Fe_{line}" / "co"       # continuous opacity path
# atm_dir = mpath / "c-0.25/MARCS_st_sph_t02_mod"
# atm = "s4750_g+1.5_m1.0_t02_x3_z-2.50_a+0.50_c-0.25_n+0.00_o+0.50_r+0.00_s+0.00.mod"
atm = home / f"{config[selected_star]["synthesis"]["model_atmosphere"]}"
marcs_original = ".false."

sspath.mkdir(parents=True, exist_ok=True)
copath.mkdir(parents=True, exist_ok=True)

#dc = "p5777_g+4.4_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00_Ba_2.27_dc.dat"
vmic = 1.0
feoh = -2.05
aoh = +0.40
Fe_abus = ["0"]

# Linelists
default_lls: list[Path] = line_list_config["master"]

mol_lls = []
for mol_ll in config[selected_star]["synthesis"]["molecular_line_lists"]:
    mol_lls.extend(ll for ll in line_list_config["molecular"][mol_ll])

effective_ll = default_lls + mol_lls
effective_ll = [Path(ll) for ll in effective_ll]
line_list_block = "\n".join(str(ll) for ll in effective_ll)

# Ba isotopic mixtures
Ba_mixtures = {
    0: "0.0242 0.0661 0.0787 0.1126 0.7184",  # Solar
    # 1: "0.0105 0.0187 0.1273 0.3289 0.5146",  # Spallation
    # 2: "0.0001 0.1325 0.0037 0.5431 0.3207",  # i process
    # 3: "0.0000 0.3924 0.0000 0.2690 0.3386",  # r process
    # 4: "0.0286 0.0222 0.0939 0.1048 0.7505",  # s process
}

# Segment file
segment = False

segments_file = working_dir / "wavelength_segments.dat"

segments = [
    (4549.033, 4559.033),  # Ba II 4554.033 Å
    (4929.077, 4939.077),  # Ba II 4934.077 Å
    (5848.675, 5858.675),  # Ba II 5853.675 Å
    (6136.713, 6146.713),  # Ba II 6141.713 Å
    (6491.898, 6501.898),  # Ba II 6496.898 Å
]

if segment:
    segments_file.write_text(
        "\n".join(f"{start} {end}" for start, end in segments) + "\n"
    )

# ---------------------------------------------------------------------------
# NLTE information file
# ---------------------------------------------------------------------------

nlte_ifn = working_dir / "DATA/nlte_infofile.dat"
nlte_ifn.parent.mkdir(parents=True, exist_ok=True)

record = f"{Z} '{elt}' 'nlte' '{model_atom}' '{dc.name}' 'ascii'"

nlte_ifn.write_text(
    "#\n"
    "# path for model atom files\n"
    f"{atom_path}/\n"
    "# path for departure files\n"
    f"{dc_path}/\n"
    f"{record}\n"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def run_program(program: Path, input_text: str, log_file: Path, append=False):
    """Run an external program with a here-document equivalent as stdin."""
    mode = "a" if append else "w"
    with log_file.open(mode) as log_handle:
        subprocess.run(
            [str(program)],
            input=input_text,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=working_dir,
        )
        # print(f"input text: \n{input_text}")


# ---------------------------------------------------------------------------
# Save Configuration
# ---------------------------------------------------------------------------

run_config = {
    "star": selected_star,
    "wavelength_range": (wmin, wmax),
    "wavelength_step": dw,
    "model_atmosphere": atm.name,
    "departure_coefficients": dc.name,
    "nlte_element": elt,
    "nlte": nltes,
    "metallicity": feoh,
    "alpha_enhancement": aoh,
    "vmic": vmic,
    "Fe_abundances": Fe_abus,
    "default_line_lists": [str(ll) for ll in default_lls],
    "molecular_line_lists": [str(ll) for ll in mol_lls],
}

with (sspath / "run_config.yaml").open("w") as f:
    yaml.dump(run_config, f)

config_save_path = sspath / "config.yaml"
with config_save_path.open("w") as f:
    yaml.dump(config[selected_star], f)

# ---------------------------------------------------------------------------
# babsma: Continuous opacity calculations
# ---------------------------------------------------------------------------


babsma_input = f"""\
###########
# wavelength range for the continuous opacity calculations. Should encompass the full
# range asked for in the following spectrum calculation (bsyn_lu)
# the step is set to 1A in babsma if smaller than 1A here.
#
'LAMBDA_MIN:'  '{wmin}'
'LAMBDA_MAX:'  '{wmax}'
'LAMBDA_STEP:' '{dw}'
###########
# model atmosphere. Various formats allowed. Only MARCS can be binary or ascii. Others are ascii
#
'MODELINPUT:' '{atm}'
'MARCS-FILE:' '{marcs_original}'
###########
# output continuous opacity file providing continuous abs and scatt at all atmospheric depths
# for a set of wavelengths defined by lambda_min/max/step. If the step is < 1A, it is set to
# 1A by default
#
'MODELOPAC:' '{copath / (atm.name + "opac")}'
###########
# Chemical composition. First overall metallicity, then alpha/Fe, Helium/H, and r- and s-process
# The latter are scaled according to their solar-system fraction (see makeabund.f)
# finally individual abundances can be provided by first giving how many of them are changed and then
# for each of them their atomic number followed by the absolute abundance on the same line
#
'METALLICITY:'    '{feoh}'
'ALPHA/Fe   :'    '{aoh}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
###########
# if xifix true, fixed microturbulence is read from next line (km/s)
# otherwise the value(s) are read from the model atmosphere.
#
'XIFIX:' 'T'
{vmic}
"""

run_program(babsma, babsma_input, log, append=False)

# ---------------------------------------------------------------------------
# bsyn (Spectral synth) + faltbon (Convolution)
# ---------------------------------------------------------------------------

for i, mixture in Ba_mixtures.items():
    isotope_values = mixture.split()

    if len(isotope_values) != 5:
        raise ValueError(f"Ba mixture {i} must contain exactly 5 isotope fractions")

    Ba_134, Ba_135, Ba_136, Ba_137, Ba_138 = isotope_values

    # for ll in lls:
    for nlte in nltes:
        for abu in Fe_abus:

            # Synthetic spectrum output filename
            # Bash calculated str/fmt_abu but did not use them in ofn.
            prefix = f"{abu}_{wmin}-{wmax}_nlte-{nlte}_"
            ofn = prefix + atm.name.replace(".int", "") + ".spec"

            print(
                f"bsyn for NLTE {nlte}, A({elt}) = {abu}, "
                f"isotopic mixture {i}"
            )

            bsyn_input = f"""\
###########
# Use NLTE if true. Source function is computed with departure coefficients
# from departure coefficient file for the atom in model atom file, if they
# are provided.
#
'NLTE :'          {nlte}
###########
# file containing NLTE information (species and associated files)
#
'NLTEINFOFILE:'   '{nlte_ifn}'
###########
# if present these files will be used to compute the spectrum in a number of
# windows specified in SEGMENTSFILE.
# Comment out if not needed, and the spectrum will be computed from
# LAMBDA_MIN to LAMBDA_MAX
# Segments must NOT overlap
#
#'SEGMENTSFILE:' '{segments_file}'
#
#'RESOLUTION:'     '1000000.'
###########
# spectral interval in the case of a single wavelength interval, i.e. no
# segmentsfile.
# min and max lambda for the calculations and constant wavelength step
#
'LAMBDA_MIN:'     '{wmin}'
'LAMBDA_MAX:'     '{wmax}'
'LAMBDA_STEP:'    '{dw}'
###########
# Intensity / Flux
#
'INTENSITY/FLUX:' 'Flux'
###########
# file containing continuous opacity at all model points. Can be computed with babsma.f
# or be a .opa file from the MARCS web site.
#
'MODELOPAC:' '{copath / (atm.name + "opac")}'
###########
# output file containing spectrum, or equivalent widths
#
'RESULTFILE :' '{sspath / ofn}'
###########
# chemical composition
#
'METALLICITY:'    '{feoh}'
'ALPHA/Fe   :'    '{aoh}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
'INDIVIDUAL ABUNDANCES:'   '1'
26  {abu}
# 26  7.46
# 22  2.00
# 24  2.30
'ISOTOPES : ' '5'
56.134 {Ba_134}
56.135 {Ba_135}
56.136 {Ba_136}
56.137 {Ba_137}
56.138 {Ba_138}
###########
# line lists. First how many there are, and then the list of lists
#
'NFILES   :' '{len(effective_ll)}'
{line_list_block}
###########
# spherical (T) or plane-parallel (F) radiative transfer.
# If spherical, a few more parameters are read
# DO NOT CHANGE THESE PARAMETERS UNLESS YOU REALLY KNOW WHAT YOU ARE DOING.
#
'SPHERICAL:'  'F'
  30
  300.00
  15
  1.30
"""

            run_program(bsyn, bsyn_input, log, append=True)

            print(ofn)

            cvl_name = ofn.replace(".spec", "_hermes.cvl")
            faltbon_input = f"""\
{sspath / ofn}
{sspath / cvl_name}
-7.5  FWHM OF CONVOLU.PROFILE=  MILLIANGSTROM, OR KM/S IF < 0.
2 PROFILE TYPE=                (1=EXP, 2=GAUSS, 3=RAD-TAN, 4=ROT)
2 FLUX-SCALE = (1=REL.FLUX,  2=ABS.FLUX))
1 NBIN = (1 = NO IN BIN )
"""

            run_program(faltbon, faltbon_input, log, append=True)

# ---------------------------------------------------------------------------
# Cleanup / organisation
# ---------------------------------------------------------------------------

for filename in ["dummy-output.dat", "radius_tau1.txt"]:
    path = working_dir / filename
    if path.exists():
        path.unlink()

shutil.copy2(log, sspath / log.name)
shutil.copy2(Path(__file__), sspath / Path(__file__).name)
if segment:
    shutil.copy2(segments_file, sspath / segments_file.name)

(sspath / "ss").mkdir(parents=True, exist_ok=True)
(sspath / "cvl").mkdir(parents=True, exist_ok=True)

for spec_file in sspath.glob("*.spec"):
    shutil.move(str(spec_file), sspath / "ss" / spec_file.name)

for cvl_file in sspath.glob("*.cvl"):
    shutil.move(str(cvl_file), sspath / "cvl" / cvl_file.name)
