import pandas as pd
import numpy as np


def load_xray(data):
    """
    Imports the csv. Renames the columns to lower letter.
    Drops the last empty column.Adds a new columns caled 'path' that has the x-rays image.png* location
    """
    df = pd.read_csv(data)
    # dropping last empty column
    df.columns = ['imgindex', 'label', 'followup', 'patientID',
                  'age', 'gender', 'viewposition', 'width',
                  'height', 'x', 'y', 'drop']
    df = df.drop(df.columns[-1], axis=1)

    df['path'] = 'data/images/'
    df['path'] = df['path'] + df['imgindex']
    return df


def create_categorical(dataframe):
    """
    Creates dummy variables through pandas and creates a 'target' column
    with a scalar vector of the dummy variables for later comparison of
    the convolutional neural networks results. Returns a dataframe.
    """
    single_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
                 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
                 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
    # create dummy variables for the labels
    for i in single_labels:
        dataframe[i] = dataframe.label.map(lambda result: 1.0 if i in result else 0)
    # creating the target vector for cross-checking with the predictions later
    # since CNN gives out as a vector of 0s and 1s
    dataframe['target'] = dataframe.apply(lambda x: [x[single_labels].values],
                                    1).map(lambda x: x[0])
    return dataframe
