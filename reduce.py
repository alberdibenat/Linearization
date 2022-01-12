import pandas as pd
import numpy as np
import random


input_file = "Gun_transverse.0175.001"
output_file = "Gun_transverse_reduced.0175.001"
bunch_dataframe = pd.read_csv(input_file,header=None, delim_whitespace = True)
bunch_dataframe.columns=['x','y','z','px','py','pz','clock','macro_charge','particle_index','status']
bunch_dataframe = bunch_dataframe[bunch_dataframe['status'] > 0]
indexes = np.array([0])
for i in range(699):
    index = int(len(bunch_dataframe)*random.random())
    indexes = np.append(indexes,index)

bunch_reduced = bunch_dataframe.take(indexes)
bunch_reduced.to_csv(output_file,sep=' ',header=None,index=None)

