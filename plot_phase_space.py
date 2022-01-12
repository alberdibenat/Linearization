import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


output_file_bunch = '3boosters_horizontal.0764.001'
bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
bunch_dataframe.columns=['x','y','z','px','py','pz','clock','macro_charge','particle_index','status']
bunch_dataframe = bunch_dataframe[bunch_dataframe['status']>=0]
delta_z = np.array(bunch_dataframe['z'].astype(float).tolist())
delta_pz = np.array(bunch_dataframe['pz'].astype(float).tolist())
px = np.array(bunch_dataframe['px'].astype(float).tolist())
py = np.array(bunch_dataframe['py'].astype(float).tolist())

z_ref = delta_z[0]
delta_z = np.delete(delta_z,0)
pz_ref = delta_pz[0]
delta_pz = np.delete(delta_pz,0)

p_ref = np.sqrt(px[0]**2+py[0]**2+pz_ref**2)
m_e = 0.511e6 #eV
gamma = np.sqrt(1+p_ref**2/m_e**2)
beta = np.sqrt(gamma**2-1)/gamma
c = 3e8



xy = np.vstack([delta_z,delta_pz])
z = gaussian_kde(xy)(xy)

#output_file_bunch2 = '3boosters.0764.001'
#bunch_dataframe2 = pd.read_csv(output_file_bunch2,header=None, delim_whitespace = True)
#bunch_dataframe2.columns=['x','y','z','px','py','pz','clock','macro_charge','particle_index','status']
#bunch_dataframe2 = bunch_dataframe2[bunch_dataframe2['status']>=0]
#delta_z2 = np.array(bunch_dataframe2['z'].astype(float).tolist())
#delta_pz2 = np.array(bunch_dataframe2['pz'].astype(float).tolist())
#px2 = np.array(bunch_dataframe2['px'].astype(float).tolist())
#py2 = np.array(bunch_dataframe2['py'].astype(float).tolist())
#
#z_ref2 = delta_z2[0]
#delta_z2 = np.delete(delta_z2,0)
#pz_ref2 = delta_pz[0]
#delta_pz2 = np.delete(delta_pz2,0)
#
#p_ref2 = np.sqrt(px2[0]**2+py2[0]**2+pz_ref2**2)
#m_e = 0.511e6 #eV
#gamma2 = np.sqrt(1+p_ref2**2/m_e**2)
#beta2 = np.sqrt(gamma2**2-1)/gamma2
#c = 3e8
#
#
#xy2 = np.vstack([delta_z2,delta_pz2])
#z2 = gaussian_kde(xy2)(xy2)



fig, ax1 = plt.subplots()
#plt.title(r'$E_{b1}$=5.00MV/m; $\phi_{b1}$=225.0º $\rightarrow $ $E_{b2}$=3.392MV/m; $\phi_{b2}$=8.399º')
#plt.title(r'$E_{b1}$=8.00MV/m; $\phi_{b1}$=-149.0º $\rightarrow $ $E_{b2}$=2.54207MV/m; $\phi_{b2}$=-82.0148º')
ax1.set_xlabel(r'$\Delta$t [ps]')
ax1.set_ylabel(r'$\Delta$pz / pz')
ax1.scatter(delta_z*1e12/beta/c,delta_pz/pz_ref,c=z,s=1)
#ax1.scatter(delta_z2*1e12/beta2/c,delta_pz2/pz_ref2,c=z2,s=1)
#ax1.set_ylim(-0.07,0.07)
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
#ax2.set_ylabel(r'# of counts', color=color)  # we already handled the x-label with ax1
#ax2.plot(bin_edges[:-1]*1e12/(beta*c), counts_delta_z, color='darkorange')
#ax2.set_yticks([])
fig.tight_layout()
#fig.savefig('Linearization_booster3_x1_x2_z=1000.pdf')
ax1.grid()
ax1.set_ylim(-0.001,0.001)
fig.savefig('Phasespace_3boosters_horizontal_1ps_noSC.png')
plt.show()
