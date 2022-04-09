# Simple Models' Builder

This app is solely for educational and research purposes.
All data should be in csv format

The web application has three major views you can choose from:
```
View Dataset

Build Model

Prediction

```

All codes can be found at **[Streamlit Model Builder](https://github.com/oigwebuike/streamlit_model_builder)**

---

**Dataset**

Numerical datasets are the target datasets for this project

There are three sets of dataset you can choose from. The first is from Seaborn, the second dataset is from SKlearn and the third is a custom dataset, which is your personal csv file.

Depending on the dataset chosen, a list of datasets are avialble to be selected to be viewed.

The pandas dataframe of the dataset is displayed by default

Three graphs are available for display, two identical graphs (with their axes inverted) are displayed by default, and a third graph is displayed when a column from the dataframe is chosen.



---

**Models**

A new model can be built using the "Build Model" view

The model to be built could be either a Classifier or Regression model, this is the first selection to be made.

Subsequently, a dataset type is selected from the list available. Right after these first two selections, the submit button ('Select Model-type') is clicked to update the available options to select from.

Depending on whether a classifier or regression option was chosen, adequate datasets options are now available.

A selection for the y-column (label or target column) is made, and also multiple-columns are chosen to be dropped, this depends on the structure of the model to be built.

---

**Predictions**

THIS CAN ONLY BE USED IF THERE IS AN ALREADY BUILT AND SAVED MODEL

Choose the data file to be predicted, and choose the model to be used

