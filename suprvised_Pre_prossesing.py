import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder




def identify_columns(df):
    num_cols = []
    cat_cols = []
    for col in df.columns.to_list():
        if df[col].dtype == "object":
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols

def check_column_data_types(df):
    
    missing_values = ["?", "N/A", "na", "--"]
    df.replace(missing_values,np.nan,inplace=True)
    
    data_type_check = {}

    for col in df.columns:
        col_values = df[col].values
        inferred_type = pd.api.types.infer_dtype(col_values)

        if pd.api.types.is_object_dtype(col_values):
            data_type_check[col] = all(isinstance(val, str) for val in col_values)
        else:
            data_type_check[col] = all(inferred_type == pd.api.types.infer_dtype(val) for val in col_values)


    return data_type_check


def convert_column_data_types(df, data_type_check):
    
    missing_values = ["?", "N/A", "na", "--"]
    df.replace(missing_values,np.nan,inplace=True)
    
    for col, is_correct in data_type_check.items():
        if not is_correct:

            col_values = df[col].values
            if all(str(val).replace("'", "").isdigit() for val in col_values if pd.notnull(val)):
                df[col] = df[col].astype(float).astype('Int64')
            else:
                if all(pd.notnull(val) and (str(val).replace("'", "").replace(".", "").isdigit() or (str(val).replace("'", "").count('.') == 1 and str(val).replace("'", "").replace(".", "", 1).isdigit())) for val in col_values):
                    df[col] = df[col].astype(float)

    return df


def Handling_NULL_Values(df,target_col):
    
    missing_values = ["?", "N/A", "na", "--"]
    df.replace(missing_values,np.nan,inplace=True)
    
    nulls_df = df.isna().sum().reset_index().rename(columns={0: "Nulls_Count"})
    nulls_df = nulls_df[nulls_df["Nulls_Count"] > 0].sort_values(
        by="Nulls_Count", ascending=False
    )
    
    nulls_df = nulls_df.set_index("index", drop=True)

    feature_to_be_dropped = []
    feature_to_be_filled = []
    rows_to_be_dropped = []   
    for col in nulls_df.index:
            if nulls_df["Nulls_Count"][col] >= df.shape[0] * (60 / 100) - 10:
                feature_to_be_dropped.append(col)
            elif nulls_df["Nulls_Count"][col] < df.shape[0] * (3 / 100) - 10:
                rows_to_be_dropped.append(col)
            else:
                feature_to_be_filled.append(col)   
                
    if target_col in feature_to_be_dropped :
        feature_to_be_dropped.remove(target_col)

    df.drop(axis=1, columns=feature_to_be_dropped, inplace=True)

    df.dropna(axis=0, subset=rows_to_be_dropped, inplace=True)

    for col in feature_to_be_filled:
        if df[col].dtype == "object":  # Check if column is categorical
            df[col] = df[col].fillna(df[col].mode()[0])  # Fill missing values with mode
        else:
            df[col] = df[col].fillna(df[col].mean().round())
            
    return df.reset_index()

# case_1

def drop_highly_uniform_columns(df,cat_col ,target_col):
    if target_col in cat_col :
        cat_col.remove(target_col)

    for col in cat_col: 
        value_counts = df[col].value_counts()
        max_count = value_counts.max()
        total_count = df.shape[0]  # Total count of all values in the DataFrame
        if max_count >= total_count * 0.8:
            df.drop(columns=col, inplace=True)  # Drop the column

    return df

# case_2

# def Handling_highly_uniform_columns(df, cat_col, target_col):
#     if target_col in cat_col:
#         cat_col.remove(target_col)

#     for col in cat_col:
#         value_counts = df[col].value_counts()
#         max_count = value_counts.max()
#         total_count = df.shape[0]  # Total count of all values in the DataFrame
#         if max_count >= total_count * 0.8:
#             if len(df[col].unique()) > 1:
#                 dummies = pd.get_dummies(df[col], prefix=col)
#                 df = pd.concat([df, dummies], axis=1)
#                 df.drop(columns=[col], inplace=True)

#     return df

# case_3

# def oversample_Handling_highly_uniform_columns(df, cat_col,target_col):
    
#     if target_col in cat_col:
#         cat_col.remove(target_col)

#     for col in cat_col: 
#         value_counts = df[col].value_counts()
#         max_count = value_counts.max()
#         total_count = df.shape[0]  # Total count of all values in the DataFrame
#         if max_count >= total_count * 0.8:
#             oversample = RandomOverSampler()
#             X_resampled, y_resampled = oversample.fit_resample(df.drop(columns=[col]), df[col])
#             df = pd.concat([X_resampled, y_resampled], axis=1)
#             break
#     return df

def drop_highly_correlated_features(df, num_cols, response):
    num_cols_without_response_variable = list(set(num_cols) - set([response]))
    numeric_df = df[num_cols_without_response_variable].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    columns_to_drop = []
    for i, col in enumerate(num_cols_without_response_variable):
        for j in range(i + 1, len(num_cols_without_response_variable)):
            other_col = num_cols_without_response_variable[j]
            if abs(corr_matrix[col][other_col]) > 0.7:
                columns_to_drop.append(other_col)

    df.drop(columns=columns_to_drop, inplace=True)
    return df

def Outliers_Handling(data, num_cols):

    for col in num_cols:
        Q1 = np.quantile(data[col], 0.25)
        Q3 = np.quantile(data[col], 0.75)
        IQR = Q3 - Q1

        Upper_Bound = Q3 + 1.5 * IQR
        Lower_Bound = Q1 - 1.5 * IQR
        
        data[col] = np.clip(data[col], Lower_Bound, Upper_Bound)

    return data

def skewness_calc(col, df):
    std_col = np.std(df[col])
    mean_col = np.mean(df[col])
    size_rows = df.shape[0]
    diff_col = df[col] - mean_col
    diff_col_powered = np.power(diff_col, 3)
    skewness = np.sum(diff_col_powered) / ((size_rows - 1) * np.power(std_col, 3))
    return skewness

def skewed_data_transformation(value, skewness):
    if abs(skewness) >= 1 and value >= 0:
        return np.log1p(value)
    else:
        return value

def min_max_scale(df, num_cols, target_col):
    
    if target_col in num_cols :
        num_cols.remove(target_col)
        
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    return df_scaled

def determine_categorical_type(column):
    unique_values = column.unique()
    num_unique_values = len(unique_values)

    if num_unique_values <= 2:
        return "nominal"
    elif num_unique_values <= 5:
        return "ordinal"
    elif num_unique_values > 5 and num_unique_values <= 20:
        if all(value.isdigit() for value in unique_values):
            return "ordinal"
        else:
            return "nominal"
    else:
        return "col_to_be_drop"

        

def encode_ordinal_columns(df, ordinal_cols):
    encoded_df = df.copy()
    label_encoder = LabelEncoder()
    for col in ordinal_cols:
        encoded_df[col] = label_encoder.fit_transform(df[col])
    return encoded_df


def encode_nominal_columns(df, nominal_cols):
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[nominal_cols])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(nominal_cols))
    df_encoded = pd.concat([df, one_hot_df], axis=1)
    # Drop the original nominal columns
    df_encoded = df_encoded.drop(nominal_cols, axis=1)

    return df_encoded


def encode_categorical_columns(df, cat_cols):
    ordinal_cols = []
    nominal_cols = []
    col_to_be_drop = []
    for col in cat_cols:
        col_type = determine_categorical_type(df[col])
        if col_type == "ordinal":
            ordinal_cols.append(col)
        elif col_type == "nominal":
            nominal_cols.append(col)
        else :
            col_to_be_drop.append(col)
            
    encoded_df = df.copy()
    if ordinal_cols:
        encoded_df = encode_ordinal_columns(encoded_df, ordinal_cols)
    if nominal_cols:
        encoded_df = encode_nominal_columns(encoded_df, nominal_cols)
    if  col_to_be_drop:
        encoded_df.drop(columns=col_to_be_drop,inplace=True)
    return encoded_df








def suprvised_preprocessing(df, lable):
    data_type_check = check_column_data_types(df)
    df = convert_column_data_types(df, data_type_check)
    
    df = Handling_NULL_Values(df,lable)


    num_cols, cat_cols = identify_columns(df)
    df = drop_highly_uniform_columns(df, cat_cols, lable)

    # num_cols, cat_cols = identify_columns(df)
    # df = Handling_highly_uniform_columns(df, cat_cols, lable)

    # num_cols, cat_cols = identify_columns(df)
    # df = oversample_Handling_highly_uniform_columns(df, cat_cols, lable)

    num_cols, cat_cols = identify_columns(df)
    df = drop_highly_correlated_features(df, num_cols, lable)

    num_cols, cat_cols = identify_columns(df)
    df = Outliers_Handling(df, num_cols)

    num_cols, cat_cols = identify_columns(df)
    for col in num_cols:
        sk = skewness_calc(col, df)
        df[col] = df[col].apply(lambda x: skewed_data_transformation(x, sk))

    num_cols, cat_cols = identify_columns(df)
    df = min_max_scale(df, num_cols, lable)


    num_cols, cat_cols = identify_columns(df)
    df = encode_categorical_columns(df, cat_cols)

    return df 


