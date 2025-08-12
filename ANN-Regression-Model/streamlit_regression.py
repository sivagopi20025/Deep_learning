import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os


#with open(path, "rb") as f:
#    label_encoder_gender = pickle.load(f)

# Load the model
model = tf.keras.models.load_model('regression_model.h5')

# load encoders and scaler 
with open("ohEncoder_geo.pkl",'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# current_dir = os.path.dirname(__file__)
# path = os.path.join(current_dir, "labelencoder_gender.pkl")
with open('labelencoder_gender.pkl','rb') as file:
    labelencoder_gender = pickle.load(file)

with open("scalerR.pkl",'rb') as file:
    scaler = pickle.load(file)

# Streamlit APP
st.title('Estimated Salary Prediction')

# User Input
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',labelencoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited',[0,1])
# estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography], 
    'Gender':[gender], 
    'Age':[age], 
    'Tenure':[tenure], 
    'Balance': [balance], 
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card], 
    'IsActiveMember':[is_active_member], 
    'Exited':[exited]
    # 'EstimatedSalary':[estimated_salary]
})
input_data.columns = input_data.columns.str.strip()

if st.button("Predict"):
    # encode gender
    input_data['Gender'] = labelencoder_gender.transform(input_data['Gender'])

    # one-hot encode geography
    geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # drop original Geography column
    input_data = input_data.drop('Geography', axis=1)
    
    # concat one-hot encoded geography columns
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

    # scale the input
    input_data_scaled = scaler.transform(input_data)

    # predict
    prediction = model.predict(input_data_scaled)
    predicted_salary = prediction[0][0]

    st.write(f"Estimated Salary: {predicted_salary:.2f}")




