import pandas as pd
def load_xray(data):
    df = pd.read_csv(data)
    # dropping last empty column
    df.columns = ['imgindex', 'label', 'followup', 'patientID',
                  'age', 'gender', 'viewposition', 'width',
                  'height', 'x', 'y', 'drop']
    df = df.drop(df.columns[-1], axis=1)

    df['path'] = 'data/images/'
    df['path'] = df['path'] + df['imgindex']
    return df
