# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('D:/AI-ML/Competition/ObesityRisk/trained_model.sav', 'rb'))

input_data = (0.274466,0.751498,0.647675,0.969308,3.000000,0.912815,0.285133,0.000000,0,1,0,0,0,2,3,2)
input_data_as_np_array = np.asarray(input_data)
input_data_reshaped = input_data_as_np_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)

mapping = {0: 'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Obesity_Type_I', 3: 'Obesity_Type_II', 4: 'Obesity_Type_III', 5: 'Overweight_Level_I', 6: 'Overweight_Level_II'}

if prediction.item() in mapping:
    bmi_category = mapping[prediction.item()]
    print('The Person is:', bmi_category)

