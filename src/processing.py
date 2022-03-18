import numpy as np
from sklearn.preprocessing import LabelEncoder as encoder


def fill_missing(train_df, test_df):
    nan_update_dict = {
        "Ever_Married": "No",
        "Graduated": "Yes",
        "Profession": "None",
        "Work_Experience": np.nanmedian(train_df["Work_Experience"]),
        "Family_Size": 1,
    }

    for col in nan_update_dict.keys():
        train_df[col].fillna(nan_update_dict[col], inplace=True)
        test_df[col].fillna(nan_update_dict[col], inplace=True)

    return train_df, test_df


def encode(train_df, test_df):
    for col in train_df.columns[:-2]:
        if train_df[col].dtype == "object":
            scaler = encoder()
            train_df[col] = scaler.fit_transform(train_df[col])
            test_df[col] = scaler.transform(test_df[col])
    return train_df, test_df
