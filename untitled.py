import streamlit as st
import seaborn as sns
import pandas as pd
import tensorflow as tf
import altair as alt

from models import Model, Model2
from sklearn import datasets
from sklearn.datasets import  load_boston, load_breast_cancer, load_diabetes, load_digits



sns_datasets = sns.get_dataset_names()
skl_load_datasets = ['load_boston', 'load_breast_cancer', 'load_diabetes', 'load_digits']


display_types = ['View Dataset', 'Build Model', 'Train-test output']
model_types = ['Classifier', 'Regression']
input_types = ['Custom Data', 'Sns Dataset', 'Skl Load Dataset']
dataset_types = ['csv', 'excel']

skl_cl_datasets = ['load_breast_cancer', 'load_digits']
sns_cl_datasets = ['anscombe', 'anagrams', 'attention', 'brain_networks', 'diamonds', 'dots', 'exercise']

sns_rg_datasets = ['mpg', 'car_crashes']
skl_rg_datasets = ['load_boston', 'load_diabetes']


view = st.sidebar.selectbox('Enter type of display to view', display_types)
cl_model = Model()
rg_model = Model2()


with st.sidebar.form(key='Datatypes'):
    input_type = st.selectbox('Select type of input data', input_types)

    st.form_submit_button('TXT')
   
    
if input_type == 'Sns Dataset':    
    with st.sidebar.form(key='Options'):
        sns_dataset = st.selectbox('Choose your dataset', sns_datasets)

        st.form_submit_button('DF')
    df = sns.load_dataset(sns_dataset)

elif input_type == 'Skl Load Dataset':    
    with st.sidebar.form(key='Options'):
        
        skl_load_data = st.selectbox('Choose your dataset', skl_load_datasets)
        skl_load_data = eval(skl_load_data)
        skl_load_data = skl_load_data()
        
        
        st.form_submit_button('DF')
    
    df = pd.DataFrame(skl_load_data.data, columns=skl_load_data.feature_names)
    y = pd.DataFrame(skl_load_data.target)
    df['y'] = y
    
else:        
    with st.sidebar.form(key='Options'):
        my_file = st.file_uploader('My file', type=dataset_types)

        st.form_submit_button('DF')

    if my_file is not None:

        file_details = {"FileName":my_file.name,"FileType":my_file.type,"FileSize":my_file.size}
        df = pd.read_csv(my_file)
    else:
        st.info('No files selected')
        df = pd.DataFrame({'A' : []}) # emoty dataframe

        
        
        
# def dataset_view():
    
#      with st.sidebar.form(key='Datatypes'):
#         st.selectbox('Select type of input data', input_types)

#         st.form_submit_button('INP')
        
        
def display_form():
    
    grph = st.empty()

    with st.sidebar.form(key='Datatypes'):
        input_type = st.selectbox('Select type of input data', input_types)

        st.form_submit_button('TXT')

    
    if input_type == 'Sns Dataset':    
        with st.sidebar.form(key='Options'):
            sns_dataset = st.selectbox('Choose your dataset', sns_datasets)

            st.form_submit_button('DF')
        df = sns.load_dataset(sns_dataset)

    elif input_type == 'Skl Load Dataset':    
        with st.sidebar.form(key='Options'):

            skl_load_data = st.selectbox('Choose your dataset', skl_load_datasets)
            skl_load_data = eval(skl_load_data)
            skl_load_data = skl_load_data()


            st.form_submit_button('DF')

        df = pd.DataFrame(skl_load_data.data, columns=skl_load_data.feature_names)
        y = pd.DataFrame(skl_load_data.target)
        df['y'] = y

    else:        
        with st.sidebar.form(key='Options'):
            my_file = st.file_uploader('My file', type=dataset_types)

            st.form_submit_button('DF')

        if my_file is not None:

            file_details = {"FileName":my_file.name,"FileType":my_file.type,"FileSize":my_file.size}
            df = pd.read_csv(my_file)
        else:
            st.info('No files selected')
            df = pd.DataFrame({'A' : []}) # emoty dataframe
            
    # with st.sidebar.form(key='Dataform'):
    
    lines = st.sidebar.multiselect('Choose lines', df.columns, default=None)        

    x = st.sidebar.selectbox('x-axis', df.columns)
    y = st.sidebar.selectbox('y-axis', df.columns)
    z = st.sidebar.selectbox('colour', df.columns)
    c1 = alt.Chart(df).mark_circle().encode(x=x, y=y, color=z)
    c2 = alt.Chart(df).mark_circle().encode(x=y, y=x, color=z) 
        
        # st.form_submit_button('VIEW')
           
    grph.line_chart(df[lines])
    st.altair_chart(c1 | c2)

    st.dataframe(df.head(50))



def display_view():
    
    grph = st.empty()
        
    if input_type == 'Sns Dataset' or 'Skl Load Datase':
        display_form()   
    else:
        if my_file is not None:
            display_form()
            st.write(file_details)
        else:
            st.info('No file uploaded')
            
            
def model_view(model1, model2):
    with st.sidebar.form(key='Choose'):
        # model_type = ['Classification', 'Regression']
        st.warning('Warning!!! Please, only use this form if you have no saved models')
        _create_mod = st.radio('Type of model to create', model_types, index=0)
        modl_name = st.text_input('Enter model name', ' ')
        
        input_type = st.selectbox('Select type of input data', input_types)

        st.form_submit_button('TXT')

        
        
        
        if df.empty:
            st.warning('Please upload a datafile')  
        else:
            
            y = st.selectbox('y-column', df.columns)
            drop_cols = st.multiselect('Drop columns', df.columns)

            #if input_type == 'Sns Dataset':
            if _create_mod == 'Classifier':
                
                if modl_name == ' ':
                    st.info('To create a model, enter a model name')

                else:
                    model1.build_class_model(modl_name, df, y, drop_cols)
            else:
                if modl_name == ' ':
                    st.info('To create a model, enter a model name')
                else:
                    model2.build_reg_model(model_name=modl_name)
        
        st.form_submit_button('Create Model')  

        
if view == 'View Dataset':
    display_view()
    
# elif view == ''
elif view == 'Build Model':
    model_view(cl_model, rg_model)
    
else:
    st.info('still under construction')
