import streamlit as st
from fastai.vision.all import *
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import plotly.express as px

#title
st.title("Qushlar rasmiga ko'ra turlarga klassifikatsiya qiluvchi model")

#rasmni joylash
file = st.file_uploader("Rasm yuklash", type=['png', 'jpeg', 'gif', 'svg', '.jfif'])
if file:
    img = PILImage.create(file)
    model1 = load_learner("add_model.pkl")
    pred, pred_id, probs = model1.predict(img)
    if pred == 'class1':
        st.subheader("Faqat qushlar rasmini klassifikatsiya qilib beradi. Iltimos qushlar rasmini yuklang!")
        st.image(file, width=224)
    else:
        st.image(file, width=224)
        #PIL convert
        img = PILImage.create(file)

        #Modelni yuklash
        model2 = load_learner("bird_model.pkl")

        #prediction
        pred, pred_id, probs = model2.predict(img)
        st.success(pred)
        st.info(f"Ehtimollik: {probs[pred_id]:.1%}")

        #plotting
        fig = px.bar(x=probs*100, 
                    y=model2.dls.vocab,
                    title="Klasslar bo'yicha ehtimollik")
        st.plotly_chart(fig)
