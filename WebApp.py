# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:08:26 2024

@author: Asus
"""

import numpy as np
import pickle
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

loaded_model = pickle.load(open('D:/AI-ML/Competition/ObesityRisk/trained_model.sav', 'rb'))

def predictions_test(input_data):
    
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)

    mapping = {0: 'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Obesity_Type_I', 3: 'Obesity_Type_II', 4: 'Obesity_Type_III', 5: 'Overweight_Level_I', 6: 'Overweight_Level_II'}

    if prediction.item() in mapping:
        bmi_category = mapping[prediction.item()]
        return bmi_category
    
    
def main():
    st.title('Obesity Predictor')
    
    from sklearn.preprocessing import MinMaxScaler


    
    Age = st.text_input('Your Age:')
    Height = st.text_input('Your Height:')
    Weight = st.text_input('Your Weight:')
    Vegetable_consumption_frequency = st.text_input('Your Vegetable_consumption_frequency:')
    Number_of_Meals = st.text_input('Your Number_of_Meals:')
    Water_Consumption = st.text_input('Your Water_Consumption:')
    Exercise_Frequency = st.text_input('Your Exercise_Frequency:')
    Screen_Time = st.text_input('Your Screen_Time')
    Gender_Female = st.text_input('Gender_Female?(0/1)')
    Line_of_Overweights_yes = st.text_input('Line_of_Overweights_yes?')
    FAVC_no = st.text_input('Your FAVC_no:')
    SMOKES_yes = st.text_input('SMOKES_yes')
    SCC_yes = st.text_input('SCC_yes')
    Alc_Consumption = st.text_input('Alc_Consumption')
    Mode_of_Transport = st.text_input('Mode_of_Transport')
    Nibbling = st.text_input('Nibbling')
    
    
    #Code
    
    obesity = ''
    
    #Creating button
    
    if st.button('Check Obesity'):
        obesity = predictions_test([Age	,Height	,Weight	,Vegetable_consumption_frequency	,Number_of_Meals	,Water_Consumption	,Exercise_Frequency,	Screen_Time	,Gender_Female	,Line_of_Overweights_yes	,FAVC_no	,SMOKES_yes,	SCC_yes,	Alc_Consumption,	Mode_of_Transport,Nibbling])
        
        
    st.success(obesity)
    


if __name__ == '__main__':
    main()
    
    














