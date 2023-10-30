import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from matplotlib import pyplot as plt
def datavis():
    data= pd.read_csv(r"C:\Users\Vinoth\Downloads\Copper_Set.xlsx - Result 1.csv")
    print(data)
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())
    data['item_date'] = pd.to_datetime(data['item_date'], format='%Y%m%d', errors='coerce').dt.date
    data['quantity tons'] = pd.to_numeric(data['quantity tons'], errors='coerce')
    data['customer'] = pd.to_numeric(data['customer'], errors='coerce')
    data['country'] = pd.to_numeric(data['country'], errors='coerce')
    data['application'] = pd.to_numeric(data['application'], errors='coerce')
    data['thickness'] = pd.to_numeric(data['thickness'], errors='coerce')
    data['width'] = pd.to_numeric(data['width'], errors='coerce')
    data['material_ref'] = data['material_ref'].str.lstrip('0')
    data['product_ref'] = pd.to_numeric(data['product_ref'], errors='coerce')
    data['delivery date'] = pd.to_datetime(data['delivery date'], format='%Y%m%d', errors='coerce').dt.date
    data['selling_price'] = pd.to_numeric(data['selling_price'], errors='coerce')
    print(data.isnull().sum())
    data['material_ref'].fillna('unknown', inplace=True)
    data=data.dropna()
    st.write(data)
    print(data.isnull().sum())
    data.to_csv(r'Data.csv')
def plot():
    data=pd.read_csv(r'Data.csv')
    col1, col2 = st.columns([1, 1], gap='large')
    col3, col4 = st.columns([1, 1], gap='large')
    col5, col6 = st.columns([1, 1], gap='large')
    col7, col8 = st.columns([1, 1], gap='large')
    col9, col10 = st.columns([1, 1], gap='large')
    with col1:
        plt.figure(figsize=(10, 5))
        x=sns.distplot(data['quantity tons'], color='red')
        plt.title('Density plot for quantity tons')
        st.pyplot(x.get_figure())
    with col2:
        plt.figure(figsize=(10, 5))
        x=sns.distplot(data['country'], color='violet')
        plt.title('Density plot for country')
        st.pyplot(x.get_figure())
    with col3:
        plt.figure(figsize=(10, 5))
        x=sns.distplot(data['application'], color='olive')
        plt.title('Density plot for application')
        st.pyplot(x.get_figure())
    with col4:
        plt.figure(figsize=(10, 5))
        x=sns.distplot(data['thickness'], color= 'brown')
        plt.title('Density plot for thickness')
        st.pyplot(x.get_figure())
    with col5:
        plt.figure(figsize=(10, 5))
        x=sns.distplot(data['width'], color='teal')
        plt.title('Density plot for width')
        st.pyplot(x.get_figure())
    with col6:
        plt.figure(figsize=(10, 5))
        x=sns.distplot(data['selling_price'], color='green')
        plt.title('Density plot for selling_price')
        st.pyplot(x.get_figure())

    mask1 = data['selling_price'] <= 0
    print(mask1.sum())
    data.loc[mask1, 'selling_price'] = np.nan

    mask1 = data['quantity tons'] <= 0
    print(mask1.sum())
    data.loc[mask1, 'quantity tons'] = np.nan

    mask1 = data['thickness'] <= 0
    print(mask1.sum())

    data.dropna(inplace=True)
    with col7:
        plt.figure(figsize=(10, 5))
        data['selling_price_log'] = np.log(data['selling_price'])
        x=sns.distplot(data['selling_price_log'])
        plt.title('Density plot for selling_price_log')
        st.pyplot(x.get_figure())
    with col8:
        plt.figure(figsize=(10, 5))
        data['thickness_log'] = np.log(data['thickness'])
        x = sns.distplot(data['thickness_log'])
        plt.title('Density plot for thickness_log')
        st.pyplot(x.get_figure())

    with col9:
        plt.figure(figsize=(10, 5))
        data['quantity tons_log'] = np.log(data['quantity tons'])
        x = sns.distplot(data['quantity tons_log'])
        plt.title('Density plot for quantity tons_log')
        st.pyplot(x.get_figure())
    with col10:
        y = data[['quantity tons_log','application','thickness_log','width','selling_price_log','country','customer','product_ref']].corr()
        plt.figure(figsize=(10, 5))
        x = sns.heatmap(y, annot=True)
        plt.title('HEAT MAP')
        st.pyplot(x.get_figure())

    data.to_csv(r'DataScaled.csv')