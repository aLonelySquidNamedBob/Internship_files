import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


spectra_dir = Path("/home/student/NICO/spectra")
synth_dir = Path("/home/student/NICO/runs/HD_115444/abundance_tests/cvl")

files = [
    # spectra_dir / "HD196944/196944.coadd", 
    # spectra_dir / "HD196944/196944.coadd.bak", 
    # spectra_dir / "HD196944/196944-coadd-3800-4500.bak", 
    # spectra_dir / "HD196944/196944-coadd-full-spectrum.bak",
    # spectra_dir / "normalized/HD_196944_normalized.coadd",
]

# file = spectra_dir / "HD196944/196944.coadd"
# synth = synth_dir / "s4750_g+1.5_m1.0_t02_x3_z-2.50_a+0.50_c-0.25_n+0.00_o+0.50_r+0.00_s+0.00.mod_4550-4560-nlte_F.spec"
# unnormalised = spectra_dir / "HD_115444.coadd"
# unnormalised = spectra_dir / "HD196944/196944.coadd.bak"
# sun = spectra_dir / "normalized/HARPS_Sun_normalized.csv"


# data = np.loadtxt(file)
# data = np.transpose(data)

# unnormalised_data = np.loadtxt(unnormalised)
# unnormalised_data = np.transpose(unnormalised_data)

# synthesised = np.loadtxt(synth)
# synthesised = np.transpose(synthesised)

# sun_data = np.loadtxt(sun, skiprows=1, delimiter=",")
# sun_data = np.transpose(sun_data)


# print("Data shape:", data.shape)
# print("Unnormalised data shape:", unnormalised_data.shape)

i = 0
for file in files:
    print(file)
    if "/normalized/" in str(file):
        data = np.loadtxt(file, skiprows=1)
        data = np.transpose(data)
        data[1], data[2] = data[2], data[1]  # swap columns
    else:
        data = np.loadtxt(file)
        data = np.transpose(data)
    print("Data shape:", data.shape)
    # remove NaN values
    mask = ~np.isnan(data[1])
    data = data[:, mask]
    print("Data shape after removing NaN values:", data.shape)
    plt.scatter(data[0], data[1], label=file.name, s=1)

    i += 1

# Plot difference between two spectra if there are exactly two files
# if len(files) == 2:
#     data1 = np.loadtxt(files[0])
#     data1 = np.transpose(data1)
#     mask1 = ~np.isnan(data1[1])
#     data1 = data1[:, mask1]

#     data2 = np.loadtxt(files[1], skiprows=1)
#     data2 = np.transpose(data2)
#     mask2 = ~np.isnan(data2[2])
#     data2 = data2[:, mask2]

#     # Interpolate the second spectrum to the wavelength grid of the first spectrum
#     # interp_flux2 = np.interp(data1[0], data2[0], data2[1])

#     # Calculate the difference
#     difference = data1[1] - data2[2]

#     print("Difference shape:", difference.shape)

#     plt.scatter(data1[0], data1[1], label=f"Original: {files[0].name}", s=1)
#     plt.scatter(data2[0], data2[2], label=f"Normalized: {files[1].name}", s=1)
#     plt.scatter(data1[0], difference * 10 + 0.8, label=f"Difference", s=1)

# # plot nan points as 0
# unnormalised_data[1][np.isnan(unnormalised_data[1])] = -1e10
# data[1][np.isnan(data[1])] = -1e10
# plt.scatter(unnormalised_data[0], unnormalised_data[1], label="HD 115444 Unnormalised", s=1)
# plt.scatter(data[0], data[1], label="HD 115444 Normalised", s=1)

# plt.plot(synthesised[0], synthesised[1], label="Synthesised")
# plt.plot(sun_data[0], sun_data[1], label="Sun")

# plt.xlim(3815.5, 3816)
# plt.xlim(4108, 4110)
# plt.ylim(0.6, 1)
# plt.legend(loc="lower left")
plt.grid(alpha=0.5)
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.savefig("/home/student/NICO/plots/plot_spectra_test.png", dpi=300)
