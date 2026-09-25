import numpy as np
import matplotlib.pyplot as plt
import os

ref_file = "runs/TIC396792499/molecular_tests/cvl/s4859_g+1.5_m1.0_t02_st_z-2.05_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00_4700-6800_nlte-T_Fe-5.40_reference_hermes.cvl"
data_ref = np.loadtxt(ref_file)
data_ref = np.transpose(data_ref)

test_files = sorted(os.listdir('runs/TIC396792499/molecular_tests/cvl'))

non_zero_residuals = []

for file in test_files:
    if file == ref_file.split('/')[-1]:
        continue  # Skip the reference file
    # print(file)

    # extract the string after ".mod" and before "_4300-4900-nlte_F_hermes.cvl"
    suffix = "_GESv5.bsyn"
    start_index = file.find("5.40_") + len("5.40_")
    end_index = file.find(suffix)
    mol_ll_name = file[start_index:end_index]

    data_test = np.loadtxt('runs/TIC396792499/molecular_tests/cvl/' + file)  # Load the first test file for comparison
    data_test = np.transpose(data_test)

    residuals = data_ref[1] - data_test[1]

    if np.abs(residuals).max() > 0.00001:
        print(f"Warning: Residuals exceed 0.00001 for {mol_ll_name}")
        non_zero_residuals.append(mol_ll_name)


    plt.figure(figsize=(10, 6))
    plt.plot(data_ref[0], data_ref[1], label="reference", linewidth=1)
    plt.plot(data_test[0], data_test[1], label="test", linewidth=1)
    plt.plot(data_ref[0], residuals * 100, label=f"residuals (x100)\nmax = {np.abs(residuals).max():.6f}", linewidth=1)

    # plt.text(0.02, 0.25, f"Residuals: max = {np.abs(residuals).max():.6f}", transform=plt.gca().transAxes, verticalalignment='top')

    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title(f"Comparison of {mol_ll_name}")
    plt.xlim(4700, 6800)
    plt.ylim(-0.5, 1.1)
    plt.grid(alpha=0.5)
    plt.legend()

    # make the directory if it doesn't exist
    os.makedirs('plots/molecular_TIC396792499', exist_ok=True)
    plt.savefig(f'plots/molecular_TIC396792499/test_plot_molecular_{mol_ll_name}.png', dpi=300)
    plt.close()

print(f"Molecular line-lists to include in the synthesis: {non_zero_residuals}")