import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder

#loading models

model=tf.keras.models.load_model('model.h5')

#load encoder and scaler

with open('label_encoder_gen.pkl','rb') as file:
    label_encode_gen=pickle.load(file)


with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)


with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## Streamlit App
st.title("Customer Churn Prediction")

## User Input

geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender=st.selectbox('Gender', label_encode_gen.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products')
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_number=st.selectbox('Is Active Number',[0,1])


#prepare input data
input_data=pd.DataFrame(
    {
  "CreditScore": [credit],
  "Gender": [label_encode_gen.transform([gender])[0]],
  "Age": [age],
  "Tenure": [tenure],
  "Balance": [balance],
  "NumOfProducts": [num_of_products],
  "HasCrCard": [has_cr_card],
  "IsActiveMember": [is_active_number],
  "EstimatedSalary": [estimated_salary],
}
)

encoder_geo=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(encoder_geo,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scaled the input data

input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)

prediction_prob=prediction[0][0]

st.write(f" Churn Probality: {prediction_prob:.2f}")
if prediction_prob> 0.5:
    st.write(" the customer will churn")
else:
    st.write(" the customer will not churn")

