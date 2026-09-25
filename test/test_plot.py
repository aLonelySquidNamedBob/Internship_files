#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# data1 = np.loadtxt('test/test1/ss/ss/s4750_g+1.5_m1.0_t02_x3_z-2.50_a+0.50_c-0.25_n+0.00_o+0.50_r+0.00_s+0.00.mod_4550-4560-nlte_F.spec')
# data1 = np.loadtxt('test/test1/ss/cvl/s4750_g+1.5_m1.0_t02_x3_z-2.50_a+0.50_c-0.25_n+0.00_o+0.50_r+0.00_s+0.00.mod_4550-4560-nlte_F_hermes.cvl')
# data1 = np.loadtxt('test/test1/ss/cvl/s4750_g+1.5_m1.0_t02_x3_z-2.50_a+0.50_c-0.25_n+0.00_o+0.50_r+0.00_s+0.00.mod_6560-6565-nlte_F_hermes.cvl')
# data2 = np.loadtxt('spectra/normalized/HD_115444_normalized.coadd')
data1 = np.loadtxt('/home/student/NICO/runs/HD_196944/abundance_tests/cvl/s5539_g+2.4_m1.0_t02_st_z-1.95_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00_5184-5204_nlte-T_Fe-5.50_hermes.cvl')
# data2 = np.loadtxt('data/normalized/HARPS_Sun_normalized.csv', skiprows=1)
data2 = np.loadtxt('/home/student/NICO/runs/HD_196944/abundance_tests/cvl/s5539_g+2.4_m1.0_t02_st_z-1.95_a+0.40_c+0.00_n+0.00_o+0.40_r+0.00_s+0.00_5184-5204_nlte-T_Fe-5.50_hermes.cvl')
data3 = np.loadtxt('/home/student/NICO/data/normalized_spectra/HD196944_normalized.coadd.csv', skiprows=1)

data1 = np.transpose(data1)
data2 = np.transpose(data2)
data3 = np.transpose(data3)

plt.figure(figsize=(20, 6))

plt.plot(data1[0], data1[1], label="NLTE 5.5")
# plt.plot(data2[0], data2[1], label="NLTE 4.5")
plt.plot(data3[0], data3[2], label="observed")

# calculare residuals by interpolating the observed spectrum to the synthetic spectrum wavelength grid
data2_interp = np.interp(data1[0], data2[0], data2[1])
residuals = data1[1] - data2_interp

# plt.plot(data1[0], residuals, label="residuals")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.xlim(5194.7-2, 5194.7+2)
# plt.xlim(4645, 4650)
# plt.xlim(5184, 5204)
# plt.xlim(6151, 6152)
# plt.xlim(4555.5, 4556.5)
# plt.ylim(-0.5, 1.1)
plt.grid(alpha=0.5)
plt.legend()

plt.savefig('test/test_plot.png', dpi=300)