import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


# In[19]:


data = load_model()

classifier = data["model"]
le_college = data["le_college"]


# In[20]:


def show_predict_page():
    st.title("HR Portal")
    st.write("""### We need some information to predict whether the Candidate will be Shortlisted or Not""")


# In[21]:


College=('T-1','T-2','T-3')
age = st.slider("Age of the candidate", 16, 50, 1)
salary = st.slider("Salary of the candidate in rupees", 0, 10000000, 100000)
Expericence = st.slider("Years of Experience", 0, 50, 3)
offer_in_hand=st.slider("Offer in Hand",0,10,0)
years_in_current_department=st.slider("Years in Current Department",0,50,1)
NumCompaniesWorked=st.slider("Years in Current Department",0,10,1)
perofExpectedhike=st.slider("% of Expected Hike",0,400,5)
College = st.selectbox("College", College)


# In[28]:


ok = st.button("Predict Shortlisting")
if ok:
    X = np.array([[age, salary,College,offer_in_hand,Expericence,years_in_current_department,NumCompaniesWorked,perofExpectedhike ]])
    X[:, 2] = le_country.transform(X[:,2])
    X = X.astype(float)
    offer = classifier.predict(X)
    st.subheader(f"The following candidate has been {offer[0]}")






 