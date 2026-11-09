from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

line_dir = Path.home() / "NICO" / "runs/HD196944/fe_abundance/20261105_151839_T5213_g1.86_v1.23_z-2.29_NLTE/lines/"

lines = os.listdir(line_dir)
lines.sort()
lines = [line_dir / l for l in lines]

selected_line = Path(lines[42])

A, chi2 = np.loadtxt(selected_line / "chi2_curve.csv", delimiter=",", skiprows=1, usecols=(0, 1), unpack=True)

coef = np.polyfit(A, chi2, 2)
A_fine = np.linspace(A.min(), A.max(), 300)
A_min = -coef[1] / (2 * coef[0])

plt.plot(A, chi2, "o-", label="trial abundances")
plt.plot(A_fine, np.polyval(coef, A_fine), label=f"parabola (min at A(Fe) = {A_min:.3f})")
# plt.xlim(5.2, 5.4)
plt.xlabel("A(Fe)")
plt.ylabel(r"$\chi^2$")
plt.title(selected_line.name)
plt.legend()
# plt.show()
plt.savefig(r"plots/chi2.png", dpi=400)


