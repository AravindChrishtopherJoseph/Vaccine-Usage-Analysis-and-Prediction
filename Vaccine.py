import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.graph_objects as go

#Streamlit Part

# Ignore PyplotGlobalUseWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

st.title("Vaccine Usage Analysis and Prediction"
            
                   )

with st.sidebar:
                       
 select=st.radio("Main Menu",["About", "Graphs", "Predictions"])

 df_en=pd.read_csv(r'EDA_vaccine.csv')

# Reading the Pickle file:
with open(r"knn_model.pkl", 'rb') as f:
    LGR = pickle.load(f)

page_be_image = f"""
<style>
[data-testid="stAppViewContainer"]{{
background-color:#F5F5F5;  /* Off-white  */
background-size: cover;
}}

[data-testid="stHeader"] {{
background-color: #FFD700;  /* Gold */
}}

[data-testid="stSidebarContent"] {{
background-color: #FFD700;  /* Gold */
}}
</style>
"""
st.markdown(page_be_image,unsafe_allow_html=True)

if select == "About":
   col1, col2 = st.columns(2) 
    
   with col1:
    st.title("H1N1 Vaccine")
    st.markdown('''The H1N1 vaccine is designed to protect against the H1N1 influenza virus, also known as the swine flu. 
                This strain of influenza caused a global pandemic in 2009, 
                leading to the development of a specific vaccine to control its spread.\n''')
    st.markdown('<p style="color:blue;">What is H1N1?</p>', unsafe_allow_html=True)
    st.markdown('''H1N1 is a subtype of the influenza A virus that primarily affects the respiratory system. 
                It was first identified in humans in the spring of 2009 and quickly spread worldwide, prompting the World Health Organization (WHO) to declare a pandemic. 
                The virus is called "swine flu" because it shares genetic similarities with flu viruses that infect pigs.\n''')
    st.markdown('<p style="color:blue;">Types of H1N1 Vaccines</p>', unsafe_allow_html=True)
    st.markdown(''' "Inactivated (Killed) Vaccine:"
                "This form of the vaccine contains an inactivated version of the H1N1 virus, which cannot cause illness but can stimulate the immune system to produce antibodies."
                "Live Attenuated Vaccine:" 
                "This is a nasal spray vaccine containing a weakened form of the virus, also designed to provoke an immune response without causing the flu."
                "Both types of vaccines were widely distributed during the 2009 pandemic".\n''')

    st.markdown('<p style="color:red;">This vaccine is to be administered only by or under the supervision of your doctor or other health care professional.</p>',unsafe_allow_html=True)

    st.header("ML Model")
    st.text("KNN model")
    st.text('''
                    .               Precision       recall       f1-score   support\n
                    0               0.86            0.91          0.88      4212\n
                    1               0.56            0.44          0.49      1130\n
                    accuracy                                      0.81      5342\n
                    macro avg       0.71            0.67          0.69      5342\n
                    weighted avg    0.80            0.81          0.80      5342''')

    

    with col2:
     image = Image.open("C:/Users/Aravind Chirshtopher/OneDrive/Desktop/Vaccine/vacimg 2.PNG")
 
     st.image(image, use_column_width=False)
      
     st.success("""
        1) h1n1_worry - Worry about the h1n1 flu(0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried.\n
        2) chronic_medic_condition - Has any chronic medical condition - (0,1).\n
        3) is_h1n1_risky	- What respondents think about the risk of getting ill with h1n1 in the absence of the vaccine- (1,2,3,4,5)- (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky).\n
        4) sick_from_h1n1_vacc - Does respondent worry about getting sick by taking the h1n1 vaccine - (1,2,3,4,5)- (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried).\n
        5) is_h1n1_vacc_effective - Does respondent think that the h1n1 vaccine is effective - (1,2,3,4,5)- (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective).\n
        6) dr_recc_h1n1_vacc	- Doctor has recommended h1n1 vaccine - (0,1).\n
        7) sex	- Respondent's sex - (Female, Male) - 1 - Female,2 - Male.\n
    """) 
     
if select == "Graphs":

    correlation_matrix = df_en.corr(method="spearman")

    # 1. Heatmap of Correlation Matrix
    fig, ax = plt.subplots(figsize=(40, 20))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt=".1f", ax=ax)
    st.title("Spearman's Rank Correlation Heatmap")
    st.pyplot(fig)

    # 2. Bar plot for H1N1 Vaccine Counts
    st.header("0 --> Didn't receive the H1N1 Vaccine, 1 --> Took the H1N1 vaccine")
    fig, ax = plt.subplots(figsize=(2,2))
    df_en["h1n1_vaccine"].value_counts(normalize=True).plot(kind="pie", ax=ax)
    st.pyplot(fig)

    # 3. Count plot of Chronic Medical Condition vs Vaccine
    st.header("0 --> People who have chronic conditions don't take vaccine, 1 --> People who do")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="chronic_medic_condition", data=df_en, hue="h1n1_vaccine", ax=ax)
    st.title("Chronic Medical Condition vs Vaccine")
    st.pyplot(fig)

    # 4. Count plot Based on Age Group
    st.title("Based on Age, people take the vaccine")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="age_bracket", data=df_en, hue="h1n1_vaccine", ax=ax)
    st.pyplot(fig)

    # 5. Count plot Based on Gender
    st.title("Based on Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="sex", data=df_en, hue="h1n1_vaccine", ax=ax)
    st.pyplot(fig)

    # 6. Box plot of H1N1 Worry
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(df_en[['h1n1_worry']], showfliers=True, ax=ax)
    st.pyplot(fig)

    # 7. Histogram of H1N1 Awareness
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df_en['h1n1_awareness'], kde=True, color='red', bins=30, ax=ax)
    st.pyplot(fig)

if select == "Predictions":

    st.title("Vaccine Usage Prediction")
    st.success("Note: This Application Will Be Helpful For Making A Best Prediction About the Recommendation of H1N1 Vaccination")

    PatientStatus=st.radio("Have you previously received the H1N1 vaccine? (Yes/No)", ('Yes','No')) 
    st.error(""" h1n1_worry - Worry about the h1n1 flu(0,1,2,3) 0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried.""")
    H1N1_Worry = st.radio("How worried are you about the H1N1 flu? (0=Not worried at all, 1=Not very worried, 2=Somewhat worried, 3=Very worried)", (0, 1, 2, 3))
    st.error("""chronic_medic_condition - Has any chronic medical condition - (0,1).""")
    chronic_medic_condition = st.radio("Do you have any chronic medical conditions? (Yes or No)", (0, 1))
    st.error("""is_h1n1_risky	- What respondents think about the risk of getting ill with h1n1 in the absence of the vaccine- (1,2,3,4,5)- (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky).""")
    is_H1N1_risky = st.radio("What do you think about the risk of getting ill with H1N1 if you don't take the vaccine? (1=Thinks it is not very low risk, 2=Thinks it is somewhat low risk, 3=don’t know if it is risky or not, 4=Thinks it is a somewhat high risk, 5=Thinks it is very highly risky)", (1, 2, 3, 4, 5))
    st.error("""sick_from_h1n1_vacc - Does respondent worry about getting sick by taking the h1n1 vaccine - (1,2,3,4,5)- (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried).""")
    sick_from_h1n1_vacc = st.radio("Are you worried about getting sick from taking the H1N1 vaccine? (1=Respondent not worried at all, 2=Respondent is not very worried, 3=Doesn't know, 4=Respondent is somewhat worried, 5=Respondent is very worried)", (1, 2, 3, 4, 5))
    st.error("""is_h1n1_vacc_effective - Does respondent think that the h1n1 vaccine is effective - (1,2,3,4,5)- (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective).""")
    is_h1n1_vacc_effective = st.radio("Do you think the H1N1 vaccine is effective? (1=Thinks not effective at all, 2=Thinks it is not very effective, 3=Doesn't know if it is effective or not, 4=Thinks it is somewhat effective, 5=Thinks it is highly effective)", (1, 2, 3, 4, 5))
    st.error("""dr_recc_h1n1_vacc	- Doctor has recommended h1n1 vaccine - (0,1).""")
    dr_recc_h1n1_vacc = st.radio("Has your doctor recommended the H1N1 vaccine? (Yes or No)", (0, 1))
    st.error("""sex	- Respondent's sex - (Female, Male) - 1 - Female,2 - Male.""")
    sex = st.radio("Select if Male (0) or Female (1)", (0,1))
    submit_button = st.button("Submit")

    if submit_button:
        input_data = pd.DataFrame({
            'h1n1_worry': [H1N1_Worry],
            'chronic_medic_condition': [chronic_medic_condition],
            'is_h1n1_risky': [is_H1N1_risky],
            'sick_from_h1n1_vacc': [sick_from_h1n1_vacc],
            'is_h1n1_vacc_effective': [is_h1n1_vacc_effective],
            'dr_recc_h1n1_vacc': [dr_recc_h1n1_vacc],
            'sex': [sex]
        })

        # Making prediction using knn
        prediction = LGR.predict(input_data)

        st.subheader("H1N1 Vaccine Recommendation")
        if prediction[0]==1:
            st.warning("The H1N1 Vaccine is recommended")
        else:
            st.success("The H1N1 Vaccine is not recommended")
        


            # (1,1,2,1,5,1,1) - 1
            # (1,1,2,1,5,1,0) - 1
            # (1,1,2,1,4,1,1) - 1
            # (1,1,2,3,4,1,1) - 1
