import os
import subprocess
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

def gaussian_fit(x,mu,sigma,a):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))
def polinomial(x,a0,a1,a2,a3):
    return a0+a1*x+a2*x**2+a3*x**3
def polinomial4(x,a0,a1,a2,a3,a4):
    return a0+a1*x+a2*x**2+a3*x**3+a4*x**4

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx,array[idx]

def getGunFunction(amplitude):
    E0_gun = amplitude
    
    origin = os.getcwd() #to save in which folder we are
    generator_file = 'generator.in'
    template_file = 'UED.template' #The template to copy
    input_file = 'UED.in' #The changed template file with our variables
    
    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@E0max@/%.3f/' %s > %s" % (E0_gun, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    
    try:
        run_generator = subprocess.run(['../../../generator', str(generator_file)], stdout = subprocess.PIPE)
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
        return 1
    except:
        print('Error during Phase-Energy curve generation')
        return 0


def getFitParams(phase,emission_time,amplitude):

    Gun_amplitude = amplitude
    run_function = getGunFunction(Gun_amplitude)

    reference_dataframe = pd.read_csv("UED.Scan.001", header=None, delim_whitespace=True)
    reference_dataframe.columns=["Scan_para","z",'FOM1','FOM2','FOM3','FOM4','FOM5','FOM6','FOM7','FOM8','FOM9','FOM10']
    phase_values = np.array(reference_dataframe['Scan_para'].astype(float).tolist())*np.pi/180 #We have to transform it to meters
    momentum_values = np.array(reference_dataframe['FOM1'].astype(float).tolist()) #already in MeV/c
    z_values = np.array(reference_dataframe['z'].astype(float).tolist()) #already in MeV/c
    m_e = 0.511 #MeV
    energy_values = np.sqrt(momentum_values**2+m_e**2) #px and py are given in eV in the file, while pz is given in MeV, so we have to transform px and py
    gamma = energy_values/m_e

    #Choose central phase for the amplitude of 20MV
    c = 3e8 #m/s
    nu = 1.3e9 #s^-1
    k = 2*np.pi*nu/c
    central_phase = phase*np.pi/180 #deg
    emission_rms_time = emission_time
    phase_slippage = 2*np.pi*nu*emission_rms_time
    fit_range = [central_phase - 3*phase_slippage,central_phase + 3*phase_slippage]
    min_index,min_phase = find_nearest(phase_values,fit_range[0])
    max_index,max_phase = find_nearest(phase_values,fit_range[1])
    phases_to_fit = phase_values[min_index:max_index]
    energies_to_fit = gamma[min_index:max_index]
    popt,pcov = curve_fit(polinomial,phases_to_fit,energies_to_fit) #3rd order poli.
    G0 = polinomial(central_phase,*popt)
    G1  = -k*(popt[1]+2*popt[2]*central_phase+3*popt[3]*central_phase**2)
    G2 = k**2*(popt[2]+3*popt[3]*central_phase)
    G3 = -k**3*popt[3]
    

    #Plot the fit to check that everything is alright
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel(r'$\phi$ [deg]')
    ax1.set_ylabel(r'$E_k$ [MeV]',color=color)
    ax1.set_title(r'$E_0$='+str(Gun_amplitude)+'MV')
    ax1.plot(phase_values*180/np.pi, gamma, color=color)
    ax1.plot(phase_values[min_index:max_index]*180/np.pi,polinomial(phase_values[min_index:max_index],*popt),color='red')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(1.0,None)
    ax1.axvline(central_phase*180/np.pi,color='black',ls='--')
    #for l in shadow_regions:
    #    ax1.axvspan(l[0], l[1], alpha=0.3, color='grey')
    ax1.axvspan(min_phase*180/np.pi, max_phase*180/np.pi, alpha=0.3, color='grey')
    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #color = 'tab:red'
    #ax2.set_ylabel(r'$z_{final}$', color=color)  # we already handled the x-label with ax1
    #ax2.plot(phase_values, z_values,'.', markersize = 1,color=color)
    #ax2.tick_params(axis='y',labelcolor=color)
    #ax2.set_ylim(0.0,None)
    plt.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('Phase_scan_fit_'+str(Gun_amplitude)+'MV.pdf')

    return G0,G1,G2,G3
    

def runAstraFunction(amplitude_booster3,phase_booster3,z_focus,phase_booster1,amplitude_booster1,phase_booster2,amplitude_booster2,phase_gun,amplitude_gun):

    focus = z_focus

    origin = os.getcwd() #to save in which folder we are
    template_file = '3boosters.template' #The template to copy
    input_file = '3boosters.in' #The changed template file with our variables


    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@cavity_phase1@/%.5f/' %s > %s" % (phase_booster1, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max1@/%.5f/' -i %s" % (amplitude_booster1, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase2@/%.5f/' -i %s" % (phase_booster2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max2@/%.5f/' -i %s" % (amplitude_booster2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@Zmax@/%.3f/' -i %s" % (focus, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase3@/%.5f/' -i %s" % (phase_booster3, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max3@/%.5f/' -i %s" % (amplitude_booster3, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@gun_phase@/%.5f/' -i %s" % (phase_gun, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max@/%.5f/' -i %s" % (amplitude_gun, input_file)
    subprocess.call(sedstr, shell=True)

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    stdout = run.stdout.decode('utf-8') #Gets the output from the completedprocess Astra
#    print('STDOUT:{}'.format(stdout)) #Prints the output in a suitable format
#    print('Iteration done, Astra run completed succesfully \n')


    #-------------------------------------------OUTPUT ANALYSIS---------------------------------------


   #We need to return a scalar variable from the function, if we want to maximize(minimize) the energy of the particles at the end of the gun then this scalar has to be the energy of the reference particle. This scalar is saved in the last line of *.ref.* by Astra.
    #All the data of the run is stored in the corresponding files gun_booster.*

    output_file_ref = '3boosters.' + str('%04d' % int(focus*100))+'.001'
    
    output_file_bunch = '3boosters.Zemit.001'
    bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
    bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
    bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
    z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
    delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
    emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

    #print('Bunch_size is: ' + str('%.3f' % (bunch_size[-1]*1e3)) + 'um' + '\n'*5)

    return bunch_size[-1] 



def runAstraFunction2(amplitude_booster3,phase_booster3,z_focus,phase_booster1,amplitude_booster1,phase_booster2,amplitude_booster2,phase_gun, amplitude_gun):

    focus = z_focus

    origin = os.getcwd() #to save in which folder we are
    template_file = '3boosters.template' #The template to copy
    input_file = '3boosters.in' #The changed template file with our variables


    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@cavity_phase1@/%.5f/' %s > %s" % (phase_booster1, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max1@/%.5f/' -i %s" % (amplitude_booster1, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase2@/%.5f/' -i %s" % (phase_booster2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max2@/%.5f/' -i %s" % (amplitude_booster2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@Zmax@/%.3f/' -i %s" % (focus, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase3@/%.5f/' -i %s" % (phase_booster3, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max3@/%.5f/' -i %s" % (amplitude_booster3, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@gun_phase@/%.5f/' -i %s" % (phase_gun, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max@/%.5f/' -i %s" % (amplitude_gun, input_file)
    subprocess.call(sedstr, shell=True)

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    stdout = run.stdout.decode('utf-8') #Gets the output from the completedprocess Astra
#    print('STDOUT:{}'.format(stdout)) #Prints the output in a suitable format
#    print('Iteration done, minimums found for '+str(amplitude_booster)+', '+str(phase_booster)+ ' \n')


    #-------------------------------------------OUTPUT ANALYSIS---------------------------------------


   #We need to return a scalar variable from the function, if we want to maximize(minimize) the energy of the particles at the end of the gun then this scalar has to be the energy of the reference particle. This scalar is saved in the last line of *.ref.* by Astra.
    #All the data of the run is stored in the corresponding files gun_booster.*

    output_file_ref = '3boosters.' + str('%04d' % int(focus*100))+'.001'
    
    output_file_bunch = '3boosters.Zemit.001'
    bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
    bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
    bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
    z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
    delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
    emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

    #print('Bunch_size is: ' + str('%.3f' % (bunch_size[-1]*1e3)) + 'um' + '\n'*5)

    return bunch_size[-1], emittance_z[-1] 





def runAstraFunction_bayes(boosters_amplitude_phase):
    
    booster1_amplitude = boosters_amplitude_phase[0]
    booster1_phase = boosters_amplitude_phase[1]
    booster2_amplitude = boosters_amplitude_phase[2]
    booster2_phase = boosters_amplitude_phase[3]
    booster3_amplitude = boosters_amplitude_phase[4]
    booster3_phase = boosters_amplitude_phase[5]

    origin = os.getcwd() #to save in which folder we are
    template_file = '3boosters_bayes.template' #The template to copy
    input_file = '3boosters_bayes.in' #The changed template file with our variables

    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@cavity_phase1@/%.5f/' %s > %s" % (booster1_phase, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max1@/%.5f/' -i %s" % (booster1_amplitude, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase2@/%.5f/' -i %s" % (booster2_phase, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max2@/%.5f/' -i %s" % (booster2_amplitude, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase3@/%.5f/' -i %s" % (booster3_phase, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max3@/%.5f/' -i %s" % (booster3_amplitude, input_file)
    subprocess.call(sedstr, shell=True)

    try:
        run = subprocess.run(['../../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    stdout = run.stdout.decode('utf-8') #Gets the output from the completedprocess Astra
#    print('STDOUT:{}'.format(stdout)) #Prints the output in a suitable format
#    print('Iteration done, Astra run completed succesfully \n')


    #-------------------------------------------OUTPUT ANALYSIS---------------------------------------


   #We need to return a scalar variable from the function, if we want to maximize(minimize) the energy of the particles at the end of the gun then this scalar has to be the energy of the reference particle. This scalar is saved in the last line of *.ref.* by Astra.
    #All the data of the run is stored in the corresponding files gun_booster.*

    
    output_file_bunch = '3boosters_bayes.Zemit.001'
    bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
    bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
    bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
    z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
    delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
    
    Ekin = np.array(bunch_dataframe['Ekin_bunch'].astype(float).tolist())
    emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

    #print('Bunch_size is: ' + str('%.3f' % (bunch_size[-1]*1e3)) + 'um' + '\n'*5)

    return bunch_size[-1] 


def runAstraCombination(phase_amplitude,z_focus,phase_gun,amplitude_gun,phase_b1,amplitude_b1,phase_b2,amplitude_b2):
    
#    print('Here!')
    focus = z_focus

    origin = os.getcwd() #to save in which folder we are
    template_file = '3boosters.template' #The template to copy
    input_file = '3boosters.in' #The changed template file with our variables


    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@cavity_phase1@/%.5f/' %s > %s" % (phase_b1, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max1@/%.5f/' -i %s" % (amplitude_b1, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase2@/%.5f/' -i %s" % (phase_b2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max2@/%.5f/' -i %s" % (amplitude_b2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@Zmax@/%.3f/' -i %s" % (focus, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase3@/%.5f/' -i %s" % (phase_amplitude[0], input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max3@/%.5f/' -i %s" % (phase_amplitude[1], input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@gun_phase@/%.5f/' -i %s" % (phase_gun, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max@/%.5f/' -i %s" % (amplitude_gun, input_file)
    subprocess.call(sedstr, shell=True)

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    stdout = run.stdout.decode('utf-8') #Gets the output from the completedprocess Astra
#    print('STDOUT:{}'.format(stdout)) #Prints the output in a suitable format
#    print('Iteration done, Astra run completed succesfully \n')


    #-------------------------------------------OUTPUT ANALYSIS---------------------------------------


   #We need to return a scalar variable from the function, if we want to maximize(minimize) the energy of the particles at the end of the gun then this scalar has to be the energy of the reference particle. This scalar is saved in the last line of *.ref.* by Astra.
    #All the data of the run is stored in the corresponding files gun_booster.*

    output_file_ref = '3boosters.' + str('%04d' % int(focus*100))+'.001'
    
    output_file_bunch = '3boosters.Zemit.001'
    bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
    bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
    bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
    z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
    delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
    emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

    #print('Bunch_size is: ' + str('%.3f' % (bunch_size[-1]*1e3)) + 'um' + '\n'*5)

    return 5*bunch_size[-1] + emittance_z[-1] 

def runAstraCombination_SC(phase_amplitude,z_focus,phase_gun,amplitude_gun,phase_b1,amplitude_b1,phase_b2,amplitude_b2):
    
#    print('Here!')
    focus = z_focus

    origin = os.getcwd() #to save in which folder we are
    template_file = '3boosters_SC.template' #The template to copy
    input_file = '3boosters_SC.in' #The changed template file with our variables


    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@cavity_phase1@/%.5f/' %s > %s" % (phase_b1, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max1@/%.5f/' -i %s" % (amplitude_b1, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase2@/%.5f/' -i %s" % (phase_b2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max2@/%.5f/' -i %s" % (amplitude_b2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@Zmax@/%.3f/' -i %s" % (focus, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase3@/%.5f/' -i %s" % (phase_amplitude[0], input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max3@/%.5f/' -i %s" % (phase_amplitude[1], input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@gun_phase@/%.5f/' -i %s" % (phase_gun, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max@/%.5f/' -i %s" % (amplitude_gun, input_file)
    subprocess.call(sedstr, shell=True)

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    stdout = run.stdout.decode('utf-8') #Gets the output from the completedprocess Astra
#    print('STDOUT:{}'.format(stdout)) #Prints the output in a suitable format
#    print('Iteration done, Astra run completed succesfully \n')


    #-------------------------------------------OUTPUT ANALYSIS---------------------------------------


   #We need to return a scalar variable from the function, if we want to maximize(minimize) the energy of the particles at the end of the gun then this scalar has to be the energy of the reference particle. This scalar is saved in the last line of *.ref.* by Astra.
    #All the data of the run is stored in the corresponding files gun_booster.*

    output_file_ref = '3boosters_SC.' + str('%04d' % int(focus*100))+'.001'
    
    output_file_bunch = '3boosters_SC.Zemit.001'
    bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
    bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
    bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
    z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
    delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
    emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

    #print('Bunch_size is: ' + str('%.3f' % (bunch_size[-1]*1e3)) + 'um' + '\n'*5)

    return 20*bunch_size[-1] + emittance_z[-1] 


def runAstraFunction_energy(booster3_phase_amplitude,z_focus,phase_booster1,amplitude_booster1,phase_booster2,amplitude_booster2):

    focus = z_focus

    origin = os.getcwd() #to save in which folder we are
    template_file = '3boosters_horizontal.template' #The template to copy
    input_file = '3boosters_horizontal.in' #The changed template file with our variables

    phase_booster3 = booster3_phase_amplitude[0]
    amplitude_booster3 = booster3_phase_amplitude[1]

    #Copy the template file and introduce our parameters
    sedstr = "sed 's/@cavity_phase1@/%.5f/' %s > %s" % (phase_booster1, template_file, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max1@/%.5f/' -i %s" % (amplitude_booster1, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase2@/%.5f/' -i %s" % (phase_booster2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max2@/%.5f/' -i %s" % (amplitude_booster2, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@Zmax@/%.3f/' -i %s" % (focus, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@cavity_phase3@/%.5f/' -i %s" % (phase_booster3, input_file)
    subprocess.call(sedstr, shell=True)
    sedstr = "sed 's/@E0max3@/%.5f/' -i %s" % (amplitude_booster3, input_file)
    subprocess.call(sedstr, shell=True)

    try:
        run = subprocess.run(['../../../Astra', str(input_file)], stdout=subprocess.PIPE)
    except:
        print('Error during Astra run')

    stdout = run.stdout.decode('utf-8') #Gets the output from the completedprocess Astra
#    print('STDOUT:{}'.format(stdout)) #Prints the output in a suitable format
#    print('Iteration done, Astra run completed succesfully \n')


    #-------------------------------------------OUTPUT ANALYSIS---------------------------------------


   #We need to return a scalar variable from the function, if we want to maximize(minimize) the energy of the particles at the end of the gun then this scalar has to be the energy of the reference particle. This scalar is saved in the last line of *.ref.* by Astra.
    #All the data of the run is stored in the corresponding files gun_booster.*

    output_file_ref = '3boosters_horizontal.' + str('%04d' % int(focus*100))+'.001'
    
    output_file_bunch = '3boosters_horizontal.Zemit.001'
    bunch_dataframe = pd.read_csv(output_file_bunch,header=None, delim_whitespace = True)
    bunch_dataframe.columns=['z_bunch','t_bunch','Ekin_bunch','bunch_size','delta_e_bunch','emittance_z_norm','z_e_derivative']
    bunch_size = np.array(bunch_dataframe['bunch_size'].astype(float).tolist())
    z_values = np.array(bunch_dataframe['z_bunch'].astype(float).tolist())
    delta_e = np.array(bunch_dataframe['delta_e_bunch'].astype(float).tolist())
    Ekin = np.array(bunch_dataframe['Ekin_bunch'].astype(float).tolist())
    emittance_z = np.array(bunch_dataframe['emittance_z_norm'].astype(float).tolist())

    #print('Bunch_size is: ' + str('%.3f' % (bunch_size[-1]*1e3)) + 'um' + '\n'*5)

    return delta_e[-1]*1000/Ekin[-1] 
