#!/usr/bin/env python3
"""Synthesise a window around one line for several Fe abundances: babsma -> bsyn -> faltbon."""

from pathlib import Path
import shutil
import subprocess
import time
import yaml

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

star_index = 4              # which star in config/stars.yaml (0, 1, 2, ...)
line = 5546.5058            # central wavelength (Å)
wmin = line - 15
wmax = line + 15
dw = 0.01                  # wavelength step (Å)

feoh = -2.21                # [Fe/H]
aoh = 0.40                  # [alpha/Fe]
vmic = 1.25                 # microturbulence (km/s)
Fe_abus = ["5.24", "-20", "5.48"]   # A(Fe) values to synthesise (-20 removes the Fe lines)
nltes = ["T"]               # "F" = LTE, "T" = NLTE
fwhm = -7.34                 # convolution FWHM: mÅ, or km/s if negative

# ---------------------------------------------------------------------------
# Paths and config files
# ---------------------------------------------------------------------------

home = Path("~/NICO").expanduser()
turbospectrum = home / "Turbospectrum_NLTE_20.1"
babsma  = turbospectrum / "exec-gf/babsma_lu"
bsyn    = turbospectrum / "exec-gf/bsyn_lu"
faltbon = turbospectrum / "Utilities/faltbon"
log     = home / "log.txt"


def load(name):
    with (home / "config" / name).open() as f:
        return yaml.safe_load(f)


stars = load("stars.yaml")
nlte_config = load("nlte.yaml")
line_list_config = load("line_lists.yaml")
reference_abundance_config = load("reference_abundance.yaml")

star = list(stars)[star_index]
synth = stars[star]["synthesis"]

atm = home / synth["model_atmosphere"]
dc = home / synth["departure_coefficients"]
elt = synth["NLTE_element"]

line_lists = list(line_list_config["master"])
if synth["molecular_line_lists"] is not None:
    for molecule in synth["molecular_line_lists"]:
        line_lists += line_list_config["molecular"][molecule]

# Every element with a measured abundance; Fe is overwritten inside the loop
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split()
ATOMIC_NUMBER = {el: z for z, el in enumerate(ELEMENTS, 1)}

reference_abundances = {
    el: entry["value"]
    for el, entry in reference_abundance_config[star]["abundances"].items()
    if entry is not None
}

# Output folders
outdir = home / "runs" / star / "abundance_tests" / f"{time.strftime('%Y%m%d_%H%M%S')}_Fe_{line}"
for sub in ["co", "ss", "cvl"]:
    (outdir / sub).mkdir(parents=True, exist_ok=True)

opac = outdir / "co" / (atm.name + "opac")

# ---------------------------------------------------------------------------
# NLTE information file: tells bsyn which model atom and departure coefficients to use
# ---------------------------------------------------------------------------

nlte_infofile = home / "DATA" / "nlte_infofile.dat"
element = nlte_config["element"][elt]
# The lone "#" first line is required: Turbospectrum's reader misparses the file if it starts with the path header
nlte_infofile.write_text(
    "#\n"
    "# path for model atom files\n"
    f"{element['model_atom_path']}/\n"
    "# path for departure files\n"
    f"{dc.parent}/\n"
    f"{element['atomic_number']} '{elt}' 'nlte' '{element['model_atom']}' '{dc.name}' 'ascii'\n"
)


def run(program, input_text):
    """Run a Turbospectrum program with input_text as its stdin, appending its output to the log."""
    with log.open("a") as f:
        subprocess.run([program], input=input_text, text=True,
                       stdout=f, stderr=subprocess.STDOUT, cwd=home, check=True)


log.write_text("")

# ---------------------------------------------------------------------------
# babsma: continuous opacity
# ---------------------------------------------------------------------------

# babsma uses a 1 Å grid and writes it rounded to 0.001 Å (5539.5058 -> 5539.506).
# If that grid starts even slightly above wmin, bsyn stops, so give it 1 Å of margin.
run(babsma, f"""\
'LAMBDA_MIN:'  '{wmin - 1}'
'LAMBDA_MAX:'  '{wmax + 1}'
'LAMBDA_STEP:' '{dw}'
'MODELINPUT:'  '{atm}'
'MARCS-FILE:'  '.false.'
'MODELOPAC:'   '{opac}'
'METALLICITY:' '{feoh}'
'ALPHA/Fe   :' '{aoh}'
'HELIUM     :' '0.00'
'R-PROCESS  :' '0.00'
'S-PROCESS  :' '0.00'
'XIFIX:' 'T'
{vmic}
""")

# ---------------------------------------------------------------------------
# bsyn (synthetic spectrum) + faltbon (convolution), one per abundance
# ---------------------------------------------------------------------------

for nlte in nltes:
    for abu in Fe_abus:
        name = f"{abu}_{wmin}-{wmax}_nlte-{nlte}_" + atm.name.replace(".int", "")
        spec = outdir / "ss" / f"{name}.spec"
        cvl = outdir / "cvl" / f"{name}_hermes.cvl"

        shift = feoh - (float(reference_abundances["Fe"]) - 7.45)
        abundances = {el: str(float(value) - shift) for el, value in reference_abundances.items()} | {"Fe": abu}
        abundance_lines = "\n".join(f"{ATOMIC_NUMBER[el]}  {float(value)}" for el, value in abundances.items())
        line_list_lines = "\n".join(str(ll) for ll in line_lists)

        print(f"bsyn: NLTE {nlte}, A(Fe) = {abu}")
        run(bsyn, f"""\
'NLTE :'          {nlte}
'NLTEINFOFILE:'   '{nlte_infofile}'
'LAMBDA_MIN:'     '{wmin}'
'LAMBDA_MAX:'     '{wmax}'
'LAMBDA_STEP:'    '{dw}'
'INTENSITY/FLUX:' 'Flux'
'MODELOPAC:'      '{opac}'
'RESULTFILE :'    '{spec}'
'METALLICITY:'    '{feoh}'
'ALPHA/Fe   :'    '{aoh}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
'INDIVIDUAL ABUNDANCES:' '{len(abundances)}'
{abundance_lines}
'ISOTOPES : ' '0'
'NFILES   :' '{len(line_lists)}'
{line_list_lines}
'SPHERICAL:'  'F'
  30
  300.00
  15
  1.30
""")

        # bsyn exits with status 0 even when it stops on an error, so check its output
        if not spec.exists() or spec.stat().st_size == 0:
            raise SystemExit(f"bsyn wrote no spectrum, see the end of {log}")

        print(f"faltbon: {cvl.name}")
        run(faltbon, f"""\
{spec}
{cvl}
{fwhm}  FWHM OF CONVOLU.PROFILE=  MILLIANGSTROM, OR KM/S IF < 0.
2 PROFILE TYPE=                (1=EXP, 2=GAUSS, 3=RAD-TAN, 4=ROT)
2 FLUX-SCALE = (1=REL.FLUX,  2=ABS.FLUX))
1 NBIN = (1 = NO IN BIN )
""")

# ---------------------------------------------------------------------------
# Keep a record of the run next to the spectra
# ---------------------------------------------------------------------------

for leftover in ["dummy-output.dat", "radius_tau1.txt"]:
    (home / leftover).unlink(missing_ok=True)

shutil.copy2(log, outdir / log.name)
shutil.copy2(__file__, outdir / Path(__file__).name)
with (outdir / "config.yaml").open("w") as f:
    yaml.dump(stars[star], f)
