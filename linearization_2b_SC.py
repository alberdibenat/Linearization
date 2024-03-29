import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from runAstra_3boosters import getGunFunction, getFitParams, runAstraFunction, runAstraFunction2, runAstraCombination, plotAstra
from scipy.optimize import minimize
from ToF_Jitter_callable import ToF_Jitter

def polinomial(x,a0,a1,a2,a3):
    return a0+a1*x+a2*x**2+a3*x**3


def emittanceForBunchMinimum(phase,amplitude,focus,B1_pha,B1_ampl,B2_pha,B2_ampl,Gun_pha,Gun_ampl,bounds_ampl,array_for_values):
    amplitude_for_focus = minimize(runAstraFunction,amplitude,args=(phase,focus,B1_pha,B1_ampl,B2_pha,B2_ampl,Gun_pha,Gun_ampl),method='SLSQP',bounds=bounds_ampl,options={'eps':0.001})
    array_for_values.append(amplitude_for_focus.x)
    size_focus, emittance = runAstraFunction2(amplitude_for_focus.x,phase,focus,B1_pha,B1_ampl,B2_pha,B2_ampl,Gun_pha,Gun_ampl) 
    return emittance


def linearizer2order(amplitude_gun,phase_gun,phase_b2,amplitude_b2):
    m_e = 0.511e6 #eV
    
    amplitude_booster1 = 8.91e6 #amplitude_b1*1e6
    phase_booster1 = 219.1*np.pi/180.0 #phase_b1*np.pi/180 #(remember that in Astra the phase is this-90deg)
    amplitude_booster2 = amplitude_b2*1e6
    phase_booster2 = phase_b2*np.pi/180 #(remember that in Astra the phase is this-90deg)
    booster3_phase_range=[-50.0,0.0]
    booster3_amplitude_range=[10.0e6,15.0e6]

    
#-----------------------------ACTIVATE THIS BLOCK FOR LINEARIZATION USING GUN PARAMETERS------------------------------------
#    G0,G1,G2,G3 = getFitParams(phase_gun,7e-12,amplitude_gun)
#    z_start = 0.31 
#---------------------------------------------------------------------------------------------------------------------------
    
#------------------------ACTIVATE THIS BLOCK FOR LINEARIZATION USING THE LONGITUDINAL PROFILE OF AN EXISTING BUNCH----------
###    input_file = "Gun_transverse_Jens.0175.001"
###    bunch_dataframe = pd.read_csv(input_file,header=None, delim_whitespace = True)
###    bunch_dataframe.columns=['x','y','z','px','py','pz','clock','macro_charge','particle_index','status']
###    
###    #Filter dataframe to only surviving particles
###    bunch_dataframe=bunch_dataframe[bunch_dataframe['status'] > 0]
###    delta_z = np.array(bunch_dataframe['z'].astype(float).tolist())
###    delta_pz = np.array(bunch_dataframe['pz'].astype(float).tolist())
###    px = np.array(bunch_dataframe['px'].astype(float).tolist())
###    py = np.array(bunch_dataframe['py'].astype(float).tolist())
###    
###    z_ref = delta_z[0]
###    pz_ref = delta_pz[0]
###    
###    delta_z[0] = 0.0 #reference particle in the center
###    pz = delta_pz + pz_ref
###    pz[0] = pz_ref
###    p = np.sqrt(pz**2 + px**2 + py**2)
###    energy = np.sqrt(p**2 + m_e**2)
###    gamma = energy/m_e
###    
###    
###    max_delta_z = np.amax(delta_z)
###    min_delta_z = np.amin(delta_z)
###    fitting_delta_z = np.linspace(min_delta_z,max_delta_z,2000)
###    popt,pcov = curve_fit(polinomial,delta_z,gamma) #3rd order poli.
###    G0,G1,G2,G3 = popt[0],popt[1],popt[2],popt[3]
###    z_start = 1.75 
    input_file = "Gun_transverse_SC.0175.001"
    bunch_dataframe = pd.read_csv(input_file,header=None, delim_whitespace = True)
    bunch_dataframe.columns = ['x','y','z','px','py','pz','clock','macro_charge','particle_index','status']
    
    #Only use surviving particles for dataframe and copy z and pz from reference part
    bunch_dataframe = bunch_dataframe[bunch_dataframe['status'] > 0]
    z_ref = bunch_dataframe.loc[0,'z'] 
    pz_ref = bunch_dataframe.loc[0,'pz'] 
    
    #Replace reference part z and pz for 0.0 and sort dataframe by z column for fitting
    bunch_dataframe.loc[0,'z'] = 0.0 
    bunch_dataframe.loc[0,'pz'] = 0.0 
    bunch_dataframe = bunch_dataframe.sort_values('z')
    
    
    delta_z = np.array(bunch_dataframe['z'].astype(float).tolist())
    delta_pz = np.array(bunch_dataframe['pz'].astype(float).tolist())
    px = np.array(bunch_dataframe['px'].astype(float).tolist())
    py = np.array(bunch_dataframe['py'].astype(float).tolist())
    pz = delta_pz + pz_ref #Total longitudinal momentum of particles
    
    
    #Convert to position instead of time:
    p = np.sqrt(pz**2 + px**2 + py**2)
    c = 3e8
    m_e = 0.511e6
    energy = np.sqrt(p**2 + m_e**2)
    gamma = energy/m_e
    beta = np.sqrt(gamma**2-1)/gamma
    
    bunch_dataframe_fit = bunch_dataframe[abs(bunch_dataframe['z']) <= 0.0015]
    #bunch_dataframe_fit = bunch_dataframe[bunch_dataframe['z'] <= 0.0015]
    
    delta_z_fit = np.array(bunch_dataframe_fit['z'].astype(float).tolist())
    delta_pz_fit = np.array(bunch_dataframe_fit['pz'].astype(float).tolist())
    px_fit = np.array(bunch_dataframe_fit['px'].astype(float).tolist())
    py_fit = np.array(bunch_dataframe_fit['py'].astype(float).tolist())
    pz_fit = delta_pz_fit + pz_ref #Total longitudinal momentum of particles
    
    
    #Convert to position instead of time:
    p_fit = np.sqrt(pz_fit**2 + px_fit**2 + py_fit**2)
    c = 3e8
    m_e = 0.511e6
    energy_fit = np.sqrt(p_fit**2 + m_e**2)
    gamma_fit = energy_fit/m_e
    beta_fit = np.sqrt(gamma_fit**2-1)/gamma_fit
    
    popt,pcov = curve_fit(polinomial,delta_z_fit,gamma_fit) #3rd order poli.
    G0,G1,G2,G3 = popt[0],popt[1],popt[2],popt[3]
    z_start = 1.75 


#---------------------------------------------------------------------------------------------------------------------------
    
    nu = 1.3e9 #s^-1
    k = 2.0*np.pi*13.0/3.0 
    wavelength = 2*np.pi/k #m
#    max_delta_z = np.amax(delta_z)
#    min_delta_z = np.amin(delta_z)
#    fitting_delta_z = np.linspace(min_delta_z,max_delta_z,2000)
#    popt,pcov = curve_fit(polinomial,delta_z,gamma) #3rd order poli.
    
    L = wavelength 
    z_booster1 = 3.2079
    z_booster2 = 4.0267
    z_booster3 = 4.8809
    z_focus = 6.0
    
    
    gamma_central = G0
    beta_central = np.sqrt(gamma_central**2-1)/gamma_central
    n11 = 1/(beta_central**2*gamma_central**3)
    n12 = (2-3*gamma_central**2)/(2*gamma_central**6*beta_central**4)
    n13 = (2-5*gamma_central**2+4*gamma_central**4)/(2*gamma_central**9*beta_central**6)
    
    
    x11 = 1+(z_booster1-z_start)*n11*G1
    x12 = (z_booster1-z_start)*(n11*G2+n12*G1**2)
    x13 = (z_booster1-z_start)*(n11*G3+2*n12*G1*G2+n13*G1**3)
    
    g0 = G0
    g1 = G1/x11
    g2 = (G2*x11-G1*x12)/x11**3
    g3 = (G3-g1*x13-2*g2*x11*x12)/x11**3
    
    
    #-----------------BOOSTER1----------------
    
    e = 1.6e-19
    me = 9.1e-31
    c=3e8
    k = 2*np.pi*13/3 #L-band 
    wavelength = 2*np.pi/k #m
    L = wavelength 
    
    alpha1 = e*amplitude_booster1/(2*me*c**2*k) 
    B10 = alpha1*k*L*np.sin(phase_booster1) #+ alpha*np.cos(phase_booster)
    B11 = -alpha1*L*k**2*np.cos(phase_booster1) #+ alpha*k*np.sin(phase_booster) 
    B12 = -alpha1*L*k**3*np.sin(phase_booster1)/2 #- k**2*alpha*np.cos(phase_booster)/2
    B13 = alpha1*L*k**4*np.cos(phase_booster1)/6 #-k**3*alpha*np.sin(phase_booster)/6 

    gamma_booster1 = g0+B10
    beta_booster1 = np.sqrt(gamma_booster1**2-1)/gamma_booster1
    n21 = 1/(beta_booster1**2*gamma_booster1**3)
    n22 = (2-3*gamma_booster1**2)/(2*gamma_booster1**6*beta_booster1**4)
    n23 = (2-5*gamma_booster1**2+4*gamma_booster1**4)/(2*gamma_booster1**9*beta_booster1**6)
    
    x21 = 1+(z_booster2-z_booster1)*(n21*(g1+B11))
    x22 = (z_booster2-z_booster1)*(n21*(g2+B12)+n22*(g1+B11)**2)
    x23 = (z_booster2-z_booster1)*(n21*(g3+B13)+2*n22*(g1+B11)*(g2+B12)+n23*(g1+B11)**3)
    
    b10 = g0+B10
    b11 = (g1+B11)/x21
    b12 = ((g2+B12)*x21 -(g1+B11)*x22)/x21**3
    b13 = ((g3+B13)-b11*x23-2*b12*x21*x22)/x21**3


    #-----------------BOOSTER2----------------
    
    e = 1.6e-19
    me = 9.1e-31
    c=3e8
    k = 2*np.pi*13/3 #L-band 
    wavelength = 2*np.pi/k #m
    L = wavelength 
    
    alpha2 = e*amplitude_booster2/(2*me*c**2*k) 
    B20 = alpha2*k*L*np.sin(phase_booster2) #+ alpha*np.cos(phase_booster)
    B21 = -alpha2*L*k**2*np.cos(phase_booster2) #+ alpha*k*np.sin(phase_booster) 
    B22 = -alpha2*L*k**3*np.sin(phase_booster2)/2 #- k**2*alpha*np.cos(phase_booster)/2
    B23 = alpha2*L*k**4*np.cos(phase_booster2)/6 #-k**3*alpha*np.sin(phase_booster)/6 
    
    gamma_booster2 = b10+B20
    beta_booster2 = np.sqrt(gamma_booster2**2-1)/gamma_booster2
    n31 = 1/(beta_booster2**2*gamma_booster2**3)
    n32 = (2-3*gamma_booster2**2)/(2*gamma_booster2**6*beta_booster2**4)
    n33 = (2-5*gamma_booster2**2+4*gamma_booster2**4)/(2*gamma_booster2**9*beta_booster2**6)
    
    x31 = 1+(z_booster3-z_booster2)*(n31*(b11+B21))
    x32 = (z_booster3-z_booster2)*(n31*(b12+B22)+n32*(b11+B21)**2)
    x33 = (z_booster3-z_booster2)*(n31*(b13+B23)+2*n32*(b11+B21)*(b12+B22)+n33*(b11+B21)**3)
    
    b20 = b10+B20
    b21 = (b11+B21)/x31
    b22 = ((b12+B22)*x31 -(b11+B21)*x32)/x31**3
    b23 = ((b13+B23)-b21*x33-2*b22*x31*x32)/x31**3

    #----------------BOOSTER3---------------------------------------------------------
    results_X1_phase = np.array([])
    results_X1_amplitude = np.array([])
    results_X2_phase = np.array([])
    results_X2_amplitude = np.array([])
    results_X3_phase = np.array([])
    results_X3_amplitude = np.array([])
    
    optimization_results_phase = np.array([])
    optimization_results_amplitude = np.array([])
    
#    n = 3000
#    for i in range(n):
#        phase_booster3 =booster3_phase_range[0]*np.pi/180 + i*(booster3_phase_range[1]-booster3_phase_range[0])/float(n)*np.pi/180
#        for j in range(n):
#            amplitude_booster3 = booster3_amplitude_range[0] + j*(booster3_amplitude_range[1]-booster3_amplitude_range[0])/float(n)
#            #Relation between E0 and alpha
#            e = 1.6e-19
#            me = 9.1e-31
#            c=3e8
#            k = 2*np.pi*13/3 #L-band 
#            wavelength = 2*np.pi/k #m
#            L = wavelength 
#            alpha3 = e*amplitude_booster3/(2*me*c**2*k) 
#            B30 = alpha3*k*L*np.sin(phase_booster3) #+ alpha*np.cos(phase_booster)
#            B31 = -alpha3*L*k**2*np.cos(phase_booster3) #+ alpha*k*np.sin(phase_booster) 
#            B32 = -alpha3*L*k**3*np.sin(phase_booster3)/2 #- k**2*alpha*np.cos(phase_booster)/2
#            B33 = alpha3*L*k**4*np.cos(phase_booster3)/6 #-k**3*alpha*np.sin(phase_booster)/6 
#            gamma_booster3 = b20+B30
#            #if (gamma_booster3<1.0):
#            #    continue
#            beta_booster3 = np.sqrt(gamma_booster3**2-1)/gamma_booster3
#            n41 = 1/(beta_booster3**2*gamma_booster3**3)
#            n42 = (2-3*gamma_booster3**2)/(2*gamma_booster3**6*beta_booster3**4)
#            n43 = (2-5*gamma_booster3**2+4*gamma_booster3**4)/(2*gamma_booster3**9*beta_booster3**6)
#            x41 = 1+(z_focus-z_booster3)*(n41*(b21+B31))
#            x42 = (z_focus-z_booster3)*(n41*(b22+B32)+n42*(b21+B31)**2)
#            x43 = (z_focus-z_booster3)*(n41*(b23+B33)+2*n42*(b21+B31)*(b22+B32)+n43*(b21+B31)**3)
#            if (abs(x41)<=0.001):
#                results_X1_phase = np.append(results_X1_phase,phase_booster3)
#                results_X1_amplitude = np.append(results_X1_amplitude,amplitude_booster3)
#            if (abs(x42)<=0.001):
#                results_X2_phase = np.append(results_X2_phase,phase_booster3)
#                results_X2_amplitude = np.append(results_X2_amplitude,amplitude_booster3)
#            if (abs(x43)<=0.1):
#                results_X3_phase = np.append(results_X3_phase,phase_booster3)
#                results_X3_amplitude = np.append(results_X3_amplitude,amplitude_booster3) 
#            if (abs(x41)<0.005 and abs(x42)<0.005):
#                optimization_results_phase = np.append(optimization_results_phase,phase_booster3)
#                optimization_results_amplitude = np.append(optimization_results_amplitude,amplitude_booster3)
    
    n = 1500
    booster3_phases = np.linspace(booster3_phase_range[0],booster3_phase_range[1],n)*np.pi/180.0
    booster3_amplitudes = np.linspace(booster3_amplitude_range[0],booster3_amplitude_range[1],n)

    phases_mesh, amplitudes_mesh = np.meshgrid(booster3_phases,booster3_amplitudes)
    phases_mesh = phases_mesh.ravel()
    amplitudes_mesh = amplitudes_mesh.ravel()
    #print(len(phases_mesh))
    #print(len(amplitudes_mesh))

    e = 1.6e-19
    me = 9.1e-31
    c=3e8
    k = 2*np.pi*13/3 #L-band 
    wavelength = 2*np.pi/k #m
    L = wavelength 
    alpha3 = e*amplitudes_mesh/(2*me*c**2*k) 
    #B0 = operate_on_Narray(phases_mesh,amplitudes_mesh, lambda a,b: e*b/(2*me*c**2*k)*k*L*np.sin(a))
    #B1 = operate_on_Narray(phases_mesh,amplitudes_mesh, lambda a,b: -e*b/(2*me*c**2*k)*k**2*L*np.cos(a))
    #B2 = operate_on_Narray(phases_mesh,amplitudes_mesh, lambda a,b: -e*b/(2*me*c**2*k)*k**3*L*np.sin(a))
    #B3 = operate_on_Narray(phases_mesh,amplitudes_mesh, lambda a,b: e*b/(2*me*c**2*k)*k**4*L*np.cos(a))
    B30 = e*amplitudes_mesh/(2*me*c**2*k)*k*L*np.sin(phases_mesh) #+ alpha*np.cos(phase_booster)
    B31 = -e*amplitudes_mesh/(2*me*c**2*k)*L*k**2*np.cos(phases_mesh) #+ alpha*k*np.sin(phase_booster) 
    B32 = -e*amplitudes_mesh/(2*me*c**2*k)*L*k**3*np.sin(phases_mesh)/2 #- k**2*alpha*np.cos(phase_booster)/2
    B33 = e*amplitudes_mesh/(2*me*c**2*k)*L*k**4*np.cos(phases_mesh)/6 #-k**3*alpha*np.sin(phase_booster)/6 
    gamma_booster3 = b20+B30
    #Check and delete any entry that may have gamma smaller than one at the end
    phases_mesh = phases_mesh[gamma_booster3>=1.0]
    amplitudes_mesh = amplitudes_mesh[gamma_booster3>=1.0]
    B30 = B30[gamma_booster3>=1.0]
    B31 = B31[gamma_booster3>=1.0]
    B32 = B32[gamma_booster3>=1.0]
    B33 = B33[gamma_booster3>=1.0]
    gamma_booster3 = gamma_booster3[gamma_booster3>=1.0]
    #print(len(B30))
    
    beta_booster3 = np.sqrt(gamma_booster3**2-1)/gamma_booster3
    n41 = 1/(beta_booster3**2*gamma_booster3**3)
    n42 = (2-3*gamma_booster3**2)/(2*gamma_booster3**6*beta_booster3**4)
    n43 = (2-5*gamma_booster3**2+4*gamma_booster3**4)/(2*gamma_booster3**9*beta_booster3**6)
    x41 = 1+(z_focus-z_booster3)*(n41*(b21+B31))
    x42 = (z_focus-z_booster3)*(n41*(b22+B32)+n42*(b21+B31)**2)
    x43 = (z_focus-z_booster3)*(n41*(b23+B33)+2*n42*(b21+B31)*(b22+B32)+n43*(b21+B31)**3)

    #Check when X1,X2 and X3 cross 0:
    idx_X1 = []
    idx_X2 = []
    idx_X3 = []
    idx_X1_tmp = np.argwhere(np.diff(np.sign(x41))).flatten()
    idx_X2_tmp = np.argwhere(np.diff(np.sign(x42))).flatten()
    idx_X3_tmp = np.argwhere(np.diff(np.sign(x43))).flatten()
    #Remove border cases (sign changes when going from the end of a line to the next one)
    for i in range(len(idx_X1_tmp)):
        if (idx_X1_tmp[i]%n != 0) and ((idx_X1_tmp[i]+1)%n != 0):
            idx_X1.append(idx_X1_tmp[i])
    for i in range(len(idx_X2_tmp)):
        if (idx_X2_tmp[i]%n != 0) and ((idx_X2_tmp[i]+1)%n != 0):
            idx_X2.append(idx_X2_tmp[i])
    for i in range(len(idx_X3_tmp)):
        if (idx_X3_tmp[i]%n != 0) and ((idx_X3_tmp[i]+1)%n != 0):
            idx_X3.append(idx_X3_tmp[i])

   
    idx_X1 = np.asarray(idx_X1)
    idx_X2 = np.asarray(idx_X2)
    idx_X3 = np.asarray(idx_X3)
    #Check when X1=X2, for that we first slightly expand the lines in which X1=0 and X2=0:
    for i in idx_X1: 
        idx_X1 = np.append(idx_X1,i-1)
        idx_X1 = np.append(idx_X1,i-2)
    #    idx_X1 = np.append(idx_X1,i-3)
    #    idx_X1 = np.append(idx_X1,i-4)
    #    idx_X1 = np.append(idx_X1,i-5)
        idx_X1 = np.append(idx_X1,i+1)
        idx_X1 = np.append(idx_X1,i+2)
    #    idx_X1 = np.append(idx_X1,i+3)
    #    idx_X1 = np.append(idx_X1,i+4)
    #    idx_X1 = np.append(idx_X1,i+5)
    for i in idx_X2: 
        idx_X2 = np.append(idx_X2,i-1)
        idx_X2 = np.append(idx_X2,i-2)
    #    idx_X2 = np.append(idx_X2,i-3)
    #    idx_X2 = np.append(idx_X2,i-4)
    #    idx_X2 = np.append(idx_X2,i-5)
        idx_X2 = np.append(idx_X2,i+1)
        idx_X2 = np.append(idx_X2,i+2)
    #    idx_X2 = np.append(idx_X2,i+3)
    #    idx_X2 = np.append(idx_X2,i+4)
    #    idx_X2 = np.append(idx_X2,i+5)
    idx_crossing =  np.intersect1d(idx_X1, idx_X2, return_indices=False)

    
    
    
    fig2,ax21 = plt.subplots()
    ax21.set_xlabel(r'$\phi_{booster}$ [deg]')
    ax21.set_ylabel(r'$E_{booster}$ [MV/m]')
#    ax21.set_title(r'$E_G$='+str(Gun_amplitude)+'MV, $\phi_G$='+str(Gun_phase)+'deg')
    if len(idx_X1)>0: ax21.scatter(phases_mesh[idx_X1]*180/np.pi, amplitudes_mesh[idx_X1], color='blue', label = r'X1$\approx$0',s=5)
    if len(idx_X2)>0: ax21.scatter(phases_mesh[idx_X2]*180/np.pi, amplitudes_mesh[idx_X2], color='darkorange', label = r'X2$\approx$0',s=5)
    if len(idx_X3)>0: ax21.scatter(phases_mesh[idx_X3]*180/np.pi, amplitudes_mesh[idx_X3], color='green', label = r'X3$\approx$0',s=5)
    if len(idx_crossing)>0: ax21.scatter(phases_mesh[idx_crossing]*180/np.pi, amplitudes_mesh[idx_crossing], color='red')# 'ro')
###    ax21.scatter(phases_mesh[idx_X1]*180/np.pi, amplitudes_mesh[idx_X1], color='blue', label = r'X1$\approx$0',s=5)
###    ax21.scatter(phases_mesh[idx_X2]*180/np.pi, amplitudes_mesh[idx_X2], color='darkorange', label = r'X2$\approx$0',s=5)
###    ax21.scatter(phases_mesh[idx_X3]*180/np.pi, amplitudes_mesh[idx_X3], color='green', label = r'X3$\approx$0',s=5)
###    ax21.scatter(phases_mesh[idx_crossing]*180/np.pi, amplitudes_mesh[idx_crossing], color='red')# 'ro')
#
    #ax21.scatter(optimization_results_phase*180/np.pi, optimization_results_amplitude, label = r'X1$\approx$0 and X2$\approx$0',color='red',s=5)
    plt.grid()
    fig2.tight_layout()
    plt.legend()
#    fig2.savefig('Solutions_'+str(Gun_phase)+'deg_'+str(Gun_amplitude)+'MV.pdf')
    plt.show()
    
    print(phases_mesh[idx_crossing]*180/np.pi)
    print(amplitudes_mesh[idx_crossing])

    #Now we will use the parametes obtained by the linearization as a seed for Astra and optimize for minimum bunch size and minimum emittance at the focus point to get the 'real' values of booster phase and amplitude, we will look in an area bounded by 10% of the values given by the analytical solution.
    
    
    Booster1_amplitude = amplitude_booster1*1e-6 
    Booster1_phase = phase_booster1*180/np.pi - 90 #We apply the necessary changes to the phase for Astra

    Booster2_amplitude = amplitude_booster2*1e-6 
    Booster2_phase = phase_booster2*180/np.pi - 90 #We apply the necessary changes to the phase for Astra

    phase_astra_seed = (np.amax(phases_mesh[idx_crossing])+np.amin(phases_mesh[idx_crossing]))*180/(np.pi*2.0) - 90
    #phase_astra_seed = (np.amax(optimization_results_phase[1])+np.amin(optimization_results_phase[0]))*180/(np.pi*2.0) - 90
    amplitude_astra_seed = (np.amax(amplitudes_mesh[idx_crossing])+np.amin(amplitudes_mesh[idx_crossing]))*1e-6/2.0



    #We will optimize for the position of the bunch minimum by varying the booster amplitude for each booster phase inside a function, this function will return the emittance and will be also optimized by looking for an emittance minimum.
    
    #Arrays to keep results
    phase_values = np.array([])
    amplitude_values = np.array([])
    bunch_size_evolution = np.array([])
    bunch_emittance_evolution = np.array([])
    focus_point_amplitudes = []

    print(phase_astra_seed,amplitude_astra_seed)

###    phase_bounds = (phase_astra_seed - 0.1*abs(phase_astra_seed), phase_astra_seed + 0.1*abs(phase_astra_seed))
    phase_bounds = [(phase_astra_seed - 0.1*abs(phase_astra_seed), phase_astra_seed + 0.1*abs(phase_astra_seed))]
###   amplitude_bounds = (amplitude_astra_seed - 0.1*amplitude_astra_seed, amplitude_astra_seed + 0.1*amplitude_astra_seed)
    amplitude_bounds = [(amplitude_astra_seed - 0.1*amplitude_astra_seed, amplitude_astra_seed + 0.1*amplitude_astra_seed)]
###    results_for_minimum = minimize(runAstraCombination, [phase_astra_seed,amplitude_astra_seed], args=(z_focus,phase_gun,amplitude_gun,Booster1_phase,Booster1_amplitude,Booster2_phase,Booster2_amplitude),method='SLSQP',bounds=[phase_bounds,amplitude_bounds],options={'eps':0.01})
    result_for_minimum = minimize(emittanceForBunchMinimum, phase_astra_seed, args=(amplitude_astra_seed,z_focus,Booster1_phase,Booster1_amplitude,Booster2_phase,Booster2_amplitude,0.0,20.0,amplitude_bounds,focus_point_amplitudes),method='SLSQP',bounds=phase_bounds,options={'eps':0.001})
    phase_for_minimal = result_for_minimum.x
    amplitude_for_minimal = focus_point_amplitudes[-1]
    minimal_size, emittance = runAstraFunction2(amplitude_for_minimal,phase_for_minimal,z_focus,Booster1_phase,Booster1_amplitude,Booster2_phase,Booster2_amplitude,phase_gun,amplitude_gun)    

    print('-------------B1 AMPLITUDE = '+ str(Booster1_amplitude) +'MV/m ---------------B1 PHASE =  '+ str(phase_booster1*180/np.pi) +'deg--------------------')
    print('-------------B2 AMPLITUDE = '+ str(Booster2_amplitude) +'MV/m ---------------B2 PHASE =  '+ str(phase_booster2*180/np.pi) +'deg--------------------')
###    print('Phase and amplitude for minimums are:                      ' + str(results_for_minimum.x[0]+90) +',                                      ' +str(results_for_minimum.x[1]))
    print('Phase and amplitude for minimums are:                      ' + str(phase_for_minimal+90) +',                                      ' +str(amplitude_for_minimal))
###    print('Minimum bunch size and emittance obtained values are:      ' + str(results_for_minimum.fun))# + ',                                          ' +str(emittance))
    print('Minimum bunch size and emittance obtained values are:      ' + str(minimal_size) + ',                                          ' +str(emittance))


###    plot_call = plotAstra(results_for_minimum.x,z_focus,phase_gun,amplitude_gun,Booster1_phase,Booster1_amplitude,Booster2_phase,Booster2_amplitude)
    plot_call = plotAstra([phase_for_minimal,amplitude_for_minimal],z_focus,phase_gun,amplitude_gun,Booster1_phase,Booster1_amplitude,Booster2_phase,Booster2_amplitude)
    ToF_class = ToF_Jitter('Results_inverted.txt')
###    jitter = ToF_class.Jitter_calculator(phase_gun, amplitude_gun, Booster1_phase, Booster1_amplitude, Booster2_phase, Booster2_amplitude, results_for_minimum.x[0], results_for_minimum.x[1])
    jitter = ToF_class.Jitter_calculator(phase_gun, amplitude_gun, Booster1_phase, Booster1_amplitude, Booster2_phase, Booster2_amplitude, phase_for_minimal, amplitude_for_minimal)
    print('Time of flight jitter is: ' + str(jitter) + 'fs')
    print('\n')


#    m = 10
#    phase_values = np.array([])
#    amplitude_values = np.array([])
#    bunch_size_evolution = np.array([])
#    bunch_emittance_evolution = np.array([])
#    for i in range(m):
#        phase = (phase_astra_seed-0.1*phase_astra_seed) + float(i)*0.2*phase_astra_seed/float(m)
#        phase_values = np.append(phase_values,phase)
#        bounds = [(amplitude_astra_seed-0.1*amplitude_astra_seed, amplitude_astra_seed+0.1*amplitude_astra_seed)]
#        res = minimize(runAstraFunction, amplitude_astra_seed, args=(phase,z_focus,Gun_phase,Gun_amplitude), method ='SLSQP', bounds=bounds, options={'eps':0.1}) #This one converges quite ok and it is bounded.
#        amplitude_for_minimal_size = res.x #Returns the amplitude value for which the bunch size is minimum at focus point
#        amplitude_values = np.append(amplitude_values, amplitude_for_minimal_size)
#        minimal_size, emittance = runAstraFunction2(amplitude_for_minimal_size,phase,z_focus,Gun_phase,Gun_amplitude)    
#        bunch_size_evolution = np.append(bunch_size_evolution,minimal_size)
#        bunch_emittance_evolution = np.append(bunch_emittance_evolution, emittance)
#    
#    #Out of all the conbinations of phase and amplitude that minimize bunch size at focus, we take the one which also minimizes emittance:
#    minimal_emittance_index = np.argmin(bunch_emittance_evolution)
#    print('-------------GUN AMPLITUDE = '+ str(Gun_amplitude) +'MV/m ---------------GUN PHASE =  '+ str(Gun_phase) +'deg--------------------')
#    print('Phase and amplitude for minimums are: ' + str(phase_values[minimal_emittance_index]) +', ' +str(amplitude_values[minimal_emittance_index]))
#    print('Minimum bunch size and emittance obtained values are: ' + str(bunch_size_evolution[minimal_emittance_index]) + ', ' +str(bunch_emittance_evolution[minimal_emittance_index]))

    return 0#minimal_size


#def emittanceForBunchMinimum(phase,amplitude,focus,Gun_pha,Gun_ampl,bounds_ampl,array_for_values):
#    amplitude_for_focus = minimize(runAstraFunction,amplitude,args(phase,focus,Gun_pha,Gun_ampl),method='SLSQP',bounds=bounds_ampl,options={'eps':0.1})
#    array_for_values=np.append(array_for_values,amplitude_for_focus.x)
#    size_focus, emittance = runAstraFunction2(amplitude_for_focus.x,phase,focus,Gun_pha,Gun_ampl) 
#    return emittance



x = np.linspace(114.5,119.5,6)
y = np.linspace(9.0,11.00,3)
X,Y = np.meshgrid(x,y)
X= X.flatten()
Y= Y.flatten()
X = [109.60]
Y = [9.87]
print(X)
for i in range(len(X)):
    get_minimal_bunch = linearizer2order(20.0,-10.0,X[i],Y[i])


