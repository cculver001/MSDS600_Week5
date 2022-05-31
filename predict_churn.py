import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filePath):
    """Loads churn data into a DataFrame from a string filePath.

    @param filePath: String represnting a local file (default NULL)
    @return: returns the DataFrame built fromt eh filePath input
    """
    df = pd.read_csv(filePath, index_col='customerID')
    return df


def make_predictions(df):
    """Uses the pycaret best model to make predictions on data in the df dataframe.

    @param df: DataFrame pre-built from a file
    @return: Prediction model based on the DataFrame
    """
    model = load_model('GBC')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No churn'},
                                            inplace=True)
    return predictions['Churn_prediction']


if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
