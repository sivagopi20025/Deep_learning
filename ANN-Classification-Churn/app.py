import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the model
model = tf.keras.models.load_model('model.h5')

# load encoders and scaler 
with open("ANN-Classification-Churn/onehot_encoder_geo.pkl",'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open("ANN-Classification-Churn/label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("ANN-Classification-Churn/scaler.pkl",'rb') as file:
    scaler = pickle.load(file)


st.title('Customer Churn Prediction')
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
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
    'EstimatedSalary':[estimated_salary]
})
input_data.columns = input_data.columns.str.strip()

if st.button("Predict"):
    # encode gender
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

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
    prediction_proba = prediction[0][0]

    st.write(f"Churn Probability: {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.markdown("The customer is likely to churn.")
    else:
        st.markdown("The customer is not likely to churn.")


