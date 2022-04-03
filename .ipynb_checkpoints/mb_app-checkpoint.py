import os
import streamlit as st
import seaborn as sns
import pandas as pd
import tensorflow as tf
import altair as alt

from models import Model, Model2
from sklearn import datasets
from sklearn.datasets import  load_boston, load_breast_cancer, load_diabetes, load_digits

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


sns_datasets = sns.get_dataset_names()
skl_load_datasets = ['load_boston', 'load_breast_cancer', 'load_diabetes', 'load_digits']

display_types = ['View Dataset', 'Build Model', 'Prediction']
model_types = ['Classifier', 'Regression']
input_types = ['Sns Dataset', 'Skl Load Dataset', 'Custom Data']

skl_cl_datasets = ['load_breast_cancer', 'load_digits']
sns_cl_datasets = ['anscombe', 'anagrams', 'attention', 'brain_networks', 'diamonds', 'dots', 'exercise']

sns_rg_datasets = ['mpg', 'car_crashes']
skl_rg_datasets = ['load_boston', 'load_diabetes']

view = st.sidebar.selectbox('Enter type of display to view', display_types)
cl_model = Model()
rg_model = Model2()


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
            my_file = st.file_uploader('My file')

            st.form_submit_button('DF')

        if my_file is not None:

            file_details = {"FileName":my_file.name,"FileType":my_file.type,"FileSize":my_file.size}
            df = pd.read_csv(my_file)
        else:
            st.info('No files selected')
            df = pd.DataFrame({'A' : []}) # emoty dataframe
              
    lines = st.sidebar.multiselect('Choose lines', df.columns, default=None)        

    x = st.sidebar.selectbox('x-axis', df.columns)
    y = st.sidebar.selectbox('y-axis', df.columns)
    z = st.sidebar.selectbox('colour', df.columns)
    c1 = alt.Chart(df).mark_circle().encode(x=x, y=y, color=z)
    c2 = alt.Chart(df).mark_circle().encode(x=y, y=x, color=z) 
        
           
    grph.line_chart(df[lines])
    st.altair_chart(c1 | c2)

    st.dataframe(df.head(50))

    
def display_view():
    
    grph = st.empty()
        
    if input_types == 'Sns Dataset' or 'Skl Load Datase':
        display_form()   
    else:
        if my_file is not None:
            display_form()
            st.write(file_details)
        else:
            st.info('No file uploaded')
            
            
def model_view(model1, model2):
    with st.sidebar.form(key='Choose'):
        
        st.warning('Warning!!! Please, only use this form if you have no saved models')
        _create_mod = st.radio('Type of model to create', model_types, index=0)
        
        input_type = st.selectbox('Select type of input data', input_types)
            
        if _create_mod == 'Classifier':
            if input_type == 'Sns Dataset':
                sns_cl_dataset = st.selectbox('Choose your dataset', sns_cl_datasets)
                df = sns.load_dataset(sns_cl_dataset)

            elif input_type == 'Skl Load Dataset':
                skl_cl_data = eval(st.selectbox('Choose your dataset', skl_cl_datasets))()

                df = pd.DataFrame(skl_cl_data.data, columns=skl_cl_data.feature_names)
                y = pd.DataFrame(skl_cl_data.target)
                df['y'] = y
            else:
                my_file = st.file_uploader('My file')

                if my_file is not None:
                    file_details = {"FileName":my_file.name,"FileType":my_file.type,"FileSize":my_file.size}
                    df = pd.read_csv(my_file)
                else:
                    st.info('No files selected')
                    df = pd.DataFrame({'A' : []}) # empty dataframe
        
        else:
            if input_type == 'Sns Dataset':
                sns_rg_dataset = st.selectbox('Choose your dataset', sns_rg_datasets)
                df = sns.load_dataset(sns_rg_dataset)

            elif input_type == 'Skl Load Dataset': 
                skl_rg_data = eval(st.selectbox('Choose your dataset', skl_rg_datasets))()

                df = pd.DataFrame(skl_rg_data.data, columns=skl_rg_data.feature_names)
                y = pd.DataFrame(skl_rg_data.target)
                df['y'] = y

            else:
                my_file = st.file_uploader('My file')

                if my_file is not None:
                    file_details = {"FileName":my_file.name,"FileType":my_file.type,"FileSize":my_file.size}
                    df = pd.read_csv(my_file)
                else:
                    st.info('No files selected')
                    df = pd.DataFrame({'A' : []}) # empty dataframe
        y = st.selectbox('y-column', df.columns)
        drop_cols = st.multiselect('Drop columns', df.columns)
    
        st.form_submit_button('MD')
                     
    with st.sidebar.form(key='Choose Model'):
        modl_name = st.text_input('Enter model name', '')
        if modl_name == '':
            st.info('To create a model, enter a model name')
        else:
            if _create_mod == 'Classifier':
                model1.build_class_model(modl_name, df, y, drop_cols)
                model1.save_model(modl_name)
            else:
                model2.build_reg_model(modl_name, df, y, drop_cols)
                

        st.form_submit_button('Create Model')       
    st.dataframe(df)
    
    
    

    
def predict_view():
    
    st.sidebar.info('select model type for prediction')
    _create_mod = st.sidebar.radio('Type of model to create', model_types, index=0)
    if _create_mod == 'Classifier':
        model = Model()
    else:
        model = Model2()
    
    my_file = st.sidebar.file_uploader('My file')
    
    with st.sidebar.form(key='ChooseFile'):
        
        if my_file is not None:
            file_details = {"FileName":my_file.name,"FileType":my_file.type,"FileSize":my_file.size}


            df = pd.read_csv(my_file)
            
            # y = st.sidebar.selectbox('Select column for prediction', df.columns)
            # y = df[y]

            del_col = st.sidebar.multiselect('Select drop column(s)', df.columns)
            df_pred = df.drop(columns=del_col)
        else:
            st.info('No files selected')
            df = pd.DataFrame({'A' : []}) # emoty dataframe   
        # model_details = {"FileName":my_model.name,"FileType":my_model.type,"FileSize":my_model.size}

        st.form_submit_button('FIL')

    my_model = st.sidebar.file_uploader('My model')
    
    
    with st.sidebar.form(key='PredictData'):
        
        if my_model is not None:
            model_details = {"FileName":my_model.name,"FileType":my_model.type,"FileSize":my_model.size}
            
            # y = df[y]
            # with st.sidebar.form(key='PredictData'):
                
            model = model.load_model(my_model.name)
            pred_y = model.predict(df_pred)
            # df['y'] = y
            df['predicted'] = pred_y
            st.write(model_details)
            # st.form_submit_button('PRD')
        
        else:
            st.info('No model selected')
            # df = pd.DataFrame({'A' : []}) # emoty dataframe
    
        st.form_submit_button('PRD')
        
    st.dataframe(df)
    
    # st.write(model_details)

    
        
if view == 'View Dataset':
    display_view()
    
# elif view == ''
elif view == 'Build Model':
    model_view(cl_model, rg_model)
    
else:
    
    predict_view()