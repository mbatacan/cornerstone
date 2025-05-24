import pandas as pd


def add_sepal_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column 'sepal_area' to the DataFrame, which is the product of 'sepal_length' and 'sepal_width'.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'sepal_length' and 'sepal_width' columns.

    Returns:
        pd.DataFrame: The DataFrame with the new 'sepal_area' column added.
    """
    df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
    return df
