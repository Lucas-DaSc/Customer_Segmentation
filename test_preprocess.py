import pandas as pd
import pytest
from train import clean_data

data = pd.DataFrame({
    "Marital_Status": ["Married", "Single"],
    "Dt_Customer": ["2012-04-15", "2014-08-23"],
    "Z_CostContact": [3, 4],
    "Z_Revenue": [11, 12],
    "Year_Birth": [1980, 1990],
    "ID": [1234, 5678],
    "Income": [30000, 50000],  
    "MntWines": [5, 10],
    "MntFruits": [2, 1],
    "MntMeatProducts": [4, 3],
    "MntFishProducts": [0, 1],
    "MntSweetProducts": [1, 0],
    "MntGoldProds": [0, 2],
    "Kidhome": [1, 0],
    "Teenhome": [0, 1],
    "Education": ["Graduation", "Master"]
})

def test_clean_data_on_minimal_df():
    cleaned = clean_data(data)

    assert "Age" in cleaned.columns
    assert "Spent" in cleaned.columns
    assert "Family_Size" in cleaned.columns
    assert "Is_Parent" in cleaned.columns
