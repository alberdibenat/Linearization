import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


output_file_bunch = '3boosters_backup.Zemit.002'
bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

m_e = 0.511 #MeV
k = 2*np.pi*13.0/3.0
L = 2*np.pi/k






fig, ax1 = plt.subplots()
color = 'tab:red'
#plt.title(r' $E_{b1}$ = 5.0MV/m, $\phi_{b1}$=62.00deg $\rightarrow$ $E_{b2}$ = 4.7789MV/m, $\phi_{b2}$=-71.3496deg')
#plt.title(r'$E_{b1}$=8.00MV/m; $\phi_{b1}$=-137.0º $\rightarrow $ $E_{b2}$=1.8403MV/m; $\phi_{b2}$=6.8471º')
#plt.title(r'$E_{b1}$=8.00MV/m; $\phi_{b1}$=-149.0º $\rightarrow $ $E_{b2}$=2.54207MV/m; $\phi_{b2}$=-82.0148º')
ax1.set_xlabel('z [m]')
ax1.set_ylabel(r'$\sigma_z$ [mm]', color=color)
ax1.plot(z_values, bunch_size, color=color)
#ax1.axvspan(0.0, 0.31, alpha=0.3, color='grey')
ax1.axvspan(3.2079-L/2, 3.2079+L/2, alpha=0.3, color='grey')
ax1.axvspan(4.0267-L/2, 4.0267+L/2, alpha=0.3, color='grey')
ax1.axvspan(4.8809-L/2, 4.8809+L/2, alpha=0.3, color='grey')
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlim(0.0,None)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel(r'$\epsilon_{z,n}$ [$\pi$ mm keV]', color=color)  # we already handled the x-label with ax1
ax2.plot(z_values, emittance_z, color=color)
ax2.axvline(10.0,ls='--',color='black')
ax2.tick_params(axis='y', labelcolor=color)
ax1.grid()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig('Bunch_length_3boosters_10MV_B1:155_B2:289.png')
plt.show()
