import streamlit as st
import catboost
from catboost import CatBoostClassifier
import PIL
from PIL import Image
st.title("Predictor of Cancer-Specific Survival for Bladder-Preserving Therapy of Muscle-Invasive Bladder Cancer")
estimator36=CatBoostClassifier()
estimator36.load_model('catboost36')
estimator60=CatBoostClassifier()
estimator60.load_model('catboost60')
with st.sidebar:
    Age = st.slider(label="Age",min_value=0,max_value=120)
    Race = st.selectbox(label = "Race:\rOther=1,Black=2,White=3",options=["1","2","3"])
    Gender = st.selectbox(label = "Gender:\rFemale=1,Male=2",options=["1","2"])
    PrimarySite = st.selectbox(label = "Primary Site:\rTrigone=0,Dome=1,Lateral=2,Anterior=3,Posterior=4",options=["0","1",'2','3','4'])
    TStage = st.selectbox(label = "T Stage",options=['2','3','4'])
    NStage = st.selectbox(label = "N Stage",options=['0','1','2'])
    MStage = st.selectbox(label = "M Stage",options=['0','1'])
    TumorSize = st.slider(label="Tumor Size mm",min_value=0,max_value=100)
    Grade = st.selectbox(label = "Grade:\rLow Grade=1,High Grade=2",options=['1','2'])
    Radiotherapy=st.selectbox(label = "Radiotherapy:\rNo\\Unkown=0,Yes=1",options=['0','1'])
    Chemotherapy=st.selectbox(label = "Chemotherapy:\rNo\\Unkown=0,Yes=1",options=['0','1'])
st.title('Real-time Prediction')
CSS36_prob = estimator36.predict_proba([Age,Race,Gender,PrimarySite,TStage,NStage,MStage,TumorSize,Grade,Radiotherapy,Chemotherapy])[1]
CSS60_prob = estimator60.predict_proba([Age,Race,Gender,PrimarySite,TStage,NStage,MStage,TumorSize,Grade,Radiotherapy,Chemotherapy])[1]

st.write('The probability of this patient dying from bladder cancer within 3 years is {:.1f}%'.format(CSS36_prob*100))
st.write('The probability of this patient dying from bladder cancer within 5 years is {:.1f}%'.format(CSS60_prob*100))

st.title('SHAP Importance')
image1=Image.open('shap_values36.png')
image2=Image.open('shap_values60.png')
st.write('3-year CSS')
st.image([image1])
st.write('5-year CSS')
st.image([image2])
