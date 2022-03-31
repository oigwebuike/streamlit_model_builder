import streamlit as st
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors as KNN


sns_datasets = sns.get_dataset_names()

model_types = ['Classifier', 'Regression']
dataset_types = ['csv', 'json', 'excel']



    
    
    