import re
import sys
import pickle
import numpy as np
import streamlit as st

from EDA import datavis, plot
from Prediction import regressorpredictbuild, classificationpredictbuild

sys.path.insert(1, r'C:\Users\Vinoth\PycharmProjects\Airbnb\venv\Lib\site-packages')
import streamlit_option_menu
st.set_page_config(layout="wide",page_title="Industrial Copper Modeling")


selected = streamlit_option_menu.option_menu("Menu", ["About", "Data","Plots and Charts",'Prediction','Contact'],
                                                 icons=["exclamation-circle","search","bar-chart","globe",'telephone-forward' ],
                                                 menu_icon= "menu-button-wide",
                                                 default_index=0,
                                                 orientation="horizontal",
                                                 styles={"nav-link": {"font-size": "15px", "text-align": "centre",  "--hover-color": "#d1798e"},
                        "nav-link-selected": {"background-color": "#b30e35"}})

if selected=="About":
    st.header('Project Title: :green[Industrial Copper Modeling]')
    st.markdown(':red[Technologies used:]')
    st.markdown(
        ':orange[Python scripting, Data Preprocessing,EDA, Streamlit, Machine learning]')
    st.markdown(':red[Domain:]')
    st.markdown(':orange[Manufacturing]')
    st.markdown(":red[About Application:]")
    st.markdown('''The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data. 
Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer . I developed a model that can predict the selling price and to predict how likely they are to become a customer I used the STATUS variable with WON being considered as Success and LOST being considered as Failure for prediction.

    ''')

if selected=='Data':
    data=datavis()
if selected=='Plots and Charts':
    plot()
if selected=='Prediction':
    tab1, tab2, tab3, tab4 = st.tabs(["REGRESSOR MODEL", "CLASSIFICATION MODEL", "PREDICT SELLING PRICE", "PREDICT STATUS"])
    with tab1:
        rm=st.button('Build Regressor model')
        if rm:
            regressorpredictbuild()
    with tab2:
        cm = st.button('Build Classification model')
        if cm:
            pass
            classificationpredictbuild()
    with tab3:

        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
                   '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
                   '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
                   '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
                   '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised',
                          'Offered', 'Offerable']
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25.,
                               67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]

        with st.form("form"):
            col1, col2, col3 = st.columns([5, 2, 5])
            with col1:
                st.write(' ')
                application = st.selectbox("Application", sorted(application_options), key=4)
                country = st.selectbox("Country", sorted(country_options), key=3)
                item_type = st.selectbox("Item Type", item_type_options, key=2)
                product_ref = st.selectbox("Product Reference", product, key=5)
                status = st.selectbox("Status", status_options, key=1)
            with col3:
                st.write(
                    f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>',
                    unsafe_allow_html=True)

                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
                st.markdown("""
                            <style>
                            div.stButton > button:first-child {
                                background-color: #820505;
                                color: white;
                                width: 100%;
                            }
                            </style>
                        """, unsafe_allow_html=True)

            flag = 0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for i in [quantity_tons, thickness, width, customer]:
                if re.match(pattern, i):
                    pass
                else:
                    flag = 1
                    break

        if submit_button and flag == 1:
            if len(i) == 0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ", i)

        if submit_button and flag == 0:

            with open(r"model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r'scaler.pkl', 'rb') as f:
                scaler_loaded = pickle.load(f)

            with open(r"t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

            with open(r"s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            newtestsample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                    country, float(customer), int(product_ref), item_type, status]])
            newtestsampleohe = t_loaded.transform(newtestsample[:, [7]]).toarray()
            newtestsampleohe2 = s_loaded.transform(newtestsample[:, [8]]).toarray()
            newtestsample = np.concatenate((newtestsample[:, [0, 1, 2, 3, 4, 5, 6, ]], newtestsampleohe, newtestsampleohe2), axis=1)
            newtestsample1 = scaler_loaded.transform(newtestsample)
            new_pred = loaded_model.predict(newtestsample1)[0]
            st.write('## :red[Predicted selling price:] ', np.exp(new_pred))

    with tab4:

        with st.form("form1"):
            col1, col2, col3 = st.columns([5, 1, 5])
            with col1:
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)")
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")


            with col3:
                st.write(' ')

                cproduct_ref = st.selectbox("Product Reference", product, key=51)
                citem_type = st.selectbox("Item Type", item_type_options, key=21)
                ccountry = st.selectbox("Country", sorted(country_options), key=31)
                capplication = st.selectbox("Application", sorted(application_options), key=41)
                csubmit_button = st.form_submit_button(label="PREDICT STATUS")

            cflag = 0
            pattern = "^(?:\d+|\d*\.\d+)$"
            for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
                if re.match(pattern, k):
                    pass
                else:
                    cflag = 1
                    break

        if csubmit_button and cflag == 1:
            if len(k) == 0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ", k)

        if csubmit_button and cflag == 0:
            with open(r"cmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            newtestsample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                    np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer),
                                    int(product_ref), citem_type]])
            newtestsampleohe = ct_loaded.transform(newtestsample[:, [8]]).toarray()
            newtestsample = np.concatenate((newtestsample[:, [0, 1, 2, 3, 4, 5, 6, 7]], newtestsampleohe), axis=1)
            newtestsample = cscaler_loaded.transform(newtestsample)
            new_pred = cloaded_model.predict(newtestsample)
            if new_pred == 1:
                st.write('## :violet[The Status is Won] ')
            else:
                st.write('## :orange[The status is Lost] ')
if selected=='Contact':
    page_bg_img = '''
        <style>
        [data-testid="stAppViewContainer"] {
        background-image: url("https://st.depositphotos.com/1038225/3793/i/600/depositphotos_37937771-stock-photo-wood-background-texture.jpg");
        background-size: cover;
        }
        </style>
        '''

    st.markdown(page_bg_img, unsafe_allow_html=True)
    col1, col2, col3 =st.columns(3)
    with col1:
        st.markdown(":violet[About me:]")
        st.markdown("Name: :orange[Vinoth Palanivel]")
        st.markdown(":green[Aspiring Data Scientist]")
        st.write("Degree: :green[Bachelor of Engineering in Electrical and Electronics Engineering]")
        st.write("E-mail: :green[vinothchennai97@gmail.com]")
        st.write("Mobile: :green[7904197698 or 9677112815]")
    with col2:
        st.markdown(":violet[Links to connect with me:]")
        st.write("Linkedin: :orange[https://www.linkedin.com/in/vinoth-palanivel-265293211/]")
        st.write("Github: :orange[https://github.com/Vinoth0208/]")
    with col3:
        st.write(":violet[Project links:]")
        st.write("1. https://github.com/Vinoth0208/Youtube_Project_For_DataScience")
        st.write("2. https://github.com/Vinoth0208/PhonepePulse")
        st.write("3. https://github.com/Vinoth0208/Bizcard")
        st.write("4. https://github.com/Vinoth0208/Airbnb")

