import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# add the title
st.title("Data Analysis Application")
st.subheader("This is simple data analysis application")

# Create a dropdown list to chose a dataset
dataset_options = ['iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox('Select a dataset', dataset_options)

# Load the selected dataset
if selected_dataset=='iris':
    df = sns.load_dataset('iris')
elif selected_dataset=='titanic':
    df = sns.load_dataset('titanic')
elif selected_dataset == 'tips':
    df = sns.load_dataset('tips')
elif selected_dataset == 'diamonds':
    df = sns.load_dataset('diamonds')

# Button to upload custom dataset
uploaded_file = st.file_uploader("Upload a custom file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) # assuming the uploaded file in csv

# Display the dataset
st.write(df)

# Display the dataset details 
st.write('Number of Rows:', df.shape[0])
st.write('Number of Columns:', df.shape[1])

# Display the column names of selected data with their data type
st.write('Column Names and Data Types', df.dtypes)

# display Missing Values if those are > 0
if df.isnull().sum().sum() > 0:
    st.write('Missing Values', df.isnull().sum().sort_values(ascending=False))
else:
    st.write('No Missing Values')

# Display the summary statistics
st.write("Summary Statistics", df.describe())

# Create a pair plot
st.subheader('Pairplot')

# Select a column to be used as hue
hue_column = st.selectbox('Select a column to be used as hue', df.columns)
st.pyplot(sns.pairplot(df, hue=hue_column))

# Create a heatmap
st.subheader('Heatmap')

# Select the columns which are numeric and then create corr_metrix
numeric_column = df.select_dtypes(include = np.number).columns
corr_matrix = df[numeric_column].corr()
numeric_column = df.select_dtypes(include = np.number).columns
corr_matrix = df[numeric_column].corr()

# from plotly import graph_objects as go

# # Create the heatmap
# heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix, y=corr_matrix,colorscale='viridis'))
# st.plotly_chart(heatmap_fig)

# # Create a heatmap
# st.subheader('Heatmap')
# # select the columns which are numeric and then create a corr_matrix
# numeric_columns = df.select_dtypes(include=np.number).columns
# corr_matrix = df[numeric_columns].corr()
# numeric_columns = df.select_dtypes(include=np.number).columns
# corr_matrix = df[numeric_columns].corr()

from plotly import graph_objects as go

# Convert the seaborn heatmap plot to a Plotly figure
heatmap_fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                       x=corr_matrix.columns,
                                       y=corr_matrix.columns,
                                       colorscale='Viridis'))
st.plotly_chart(heatmap_fig)