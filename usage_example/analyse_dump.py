import plonk
import CaSh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

print("Loading h5...")
try:
    dump = CaSh.h5_Dump("triple80_00010.h5")#triple80_01000.h5")#HD-2_01500.h5")#
except FileNotFoundError:
    down_link = "https://unimi2013-my.sharepoint.com/:f:/g/personal/simone_ceppi_unimi_it/Eovpo6zslC9PojWizkh7fNUBays_bygI1SCayWwo4HV4Jg"
    print("Download .h5 test dumps at",link)

print("Computing arrays...")
dump.compute_arrays()

print("Binning and computing quantities in each semi-major axis bin...")
dump.compute_ring_average_e(1,30,30) #fills a_bins and a_peric (eccentricity vector)

print("Estimating a and e from snap profiles...")

dump.compute_cavity_semimajor_axis(slices=32)

aa = dump.cavity_semimajor_axis
ee = dump.cavity_eccentricity

print("profile a:", aa)
print("profile e:", ee)

print("Computing cavity orbital parameters and shape...")
orbit_xyz = dump.compute_cavity_shape()

nn=dump.cav_idx
a=dump.cav_a

ecc=dump.cav_e


print("Plotting snap...")
units = {'position': 'au', 'density': 'g/cm^3', 'projection': 'cm'}
ax=dump.snap.image(quantity='density',
              extent=(-50, 50, -50, 50),
              units=units,
              cmap='gist_heat',
                    )

ax.figure.set_size_inches(12, 10)

print("Plotting cavity ellipse...")

ax.plot(orbit_xyz[:,0], orbit_xyz[:,1])

print("h, e:",dump.a_h[nn],dump.a_peric[nn])
print("omega, i, Omega:",np.degrees(dump.cav_omega),np.degrees(dump.cav_i),np.degrees(dump.cav_Omega))
print("ecc, a:",ecc,a)

print("Plotting...")
plt.show()
