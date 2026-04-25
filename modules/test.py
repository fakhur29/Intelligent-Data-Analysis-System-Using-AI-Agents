import pandas as pd
# from autoML import select_best_model
# import numpy as np

# from data_cleaner import encode_categorical

# from data_cleaner import scale_features

# from data_cleaner import drop_irrelevant_columns

# from eda import get_basic_info



############# format_correction##############################
# # Import your function
# from data_cleaner import format_correction

# # 1. Create a "Messy" dataset for testing
# data = {
#     'City': [' new york', 'NEW YORK', 'Chicago ', 'chicago'],
#     'Salary': ['$5,000', '$4,200', '3500', '$6,000'],
#     'Gender': [' male', 'FEMALE', 'Female', 'MALE ']
# }
# df_test = pd.DataFrame(data)

# print("--- Data BEFORE Format Correction ---")
# print(df_test)
# print("\n")
# print("\nUnique values in 'City' after cleaning:", df_test['City'].unique())

# # 2. Run the function
# df_cleaned = format_correction(df_test)

# print("--- Data AFTER Format Correction ---")
# print(df_cleaned)

# # 3. Verification Check
# print("\nUnique values in 'City' after cleaning:", df_cleaned['City'].unique())

############################ fix data types ##################################
# # Import your function
# from data_cleaner import fix_data_types

# # 1. Create a dataset where everything is currently an 'object' (string)
# data = {
#     'Student_ID': ['101', '102', '103'],      # Numbers as strings
#     'GPA': ['3.8', '3.5', '3.9'],             # Floats as strings
#     'Enrollment_Date': ['2023-01-15', '2023-02-10', '2023-03-05'], # Dates as strings
#     'Name': ['Ali', 'Ahmed', 'Sara']          # Actual text strings
# }

# df_test = pd.DataFrame(data)

# print("--- Data Types BEFORE ---")
# print(df_test.dtypes)

# # 2. Run the function
# df_fixed = fix_data_types(df_test)

# print("\n--- Data Types AFTER ---")
# print(df_fixed.dtypes)

# # 3. Quick Math Test
# # This would fail before the fix because you can't find the mean of strings
# print("\nAverage GPA:", df_fixed['GPA'].mean())

##################outlier deteector #################################


# from data_cleaner import detect_outliers

# # 1. Create test data (Mostly ages 20-30, with one '2' and one '150')
# data = {
#     'Name': ['Ali', 'Ahmed', 'Sara', 'Zain', 'Khan', 'Hassan'],
#     'Age': [22, 25, 28, 150, 24, 2], 
#     'Score': [85, 88, 90, 82, 300, 87] # 300 is an outlier for a test score
# }

# df_test = pd.DataFrame(data)

# print("--- Test Dataset ---")
# print(df_test)

# # 2. Run the detection
# report = detect_outliers(df_test)

# # 3. Print the report
# print("\n--- Outlier Report ---")
# if not report:
#     print("No outliers detected.")
# else:
#     for column, count in report.items():
#         print(f"Column '{column}': Found {count} outliers")

####################### handle outlier ##########################################


# # Import your functions
# from data_cleaner import detect_outliers, handle_outliers

# # 1. Create test data with obvious outliers
# data = {
#     'Age': [20, 21, 22, 23, 24, 150], # 150 is an outlier
#     'Score': [80, 82, 85, 88, 90, 10]  # 10 is an outlier
# }
# df_test = pd.DataFrame(data)

# print(f"Original Row Count: {len(df_test)}")

# # 2. Run the removal function
# df_cleaned = handle_outliers(df_test)

# print(f"Cleaned Row Count: {len(df_cleaned)}")

# # 3. Final Verification
# print("\n--- Remaining Data ---")
# print(df_cleaned)

# if len(df_cleaned) < len(df_test):
#     print("\nSUCCESS: Outliers were removed.")
############################### encode categorical ##############################
# # 1. Create a test dataset
# data = {
#     'City': ['Karachi', 'Lahore', 'Karachi', 'Islamabad', 'Lahore', 'Karachi'], # 3 unique values (Good)
#     'Full_Name': ['Ali Khan', 'Sara Ahmed', 'Zain Malik', 'Hassan Raza', 'Fatimah Ali', 'Bilal Shah'], # 6 unique values (Too many for our test)
#     'Score': [85, 90, 88, 76, 92, 81] # Numeric column (Should be ignored by this function)
# }

# df_test = pd.DataFrame(data)

# print("--- Original Data ---")
# print(df_test)
# print(f"Columns: {df_test.columns.tolist()}")

# # 2. Run the function with a low threshold (e.g., 4)
# # Since 'Full_Name' has 6 unique values, it should be dropped.
# # Since 'City' has 3 unique values, it should be encoded.
# df_processed = encode_categorical(df_test, threshold=4)

# print("\n--- Processed Data ---")
# print(df_processed)
# print(f"Remaining Columns: {df_processed.columns.tolist()}")

# # 3. Verification Logic
# if 'Full_Name' not in df_processed.columns:
#     print("\nSUCCESS: 'Full_Name' was dropped due to high cardinality.")
# if df_processed['City'].dtype != 'object':
#     print("SUCCESS: 'City' was successfully encoded into numbers.")

########################################## scale features ##############################
# data = {
#     'Age': [20, 30, 40, 50, 60],               # Continuous numbers
#     'Salary': [40000, 50000, 60000, 70000, 80000], # Large continuous numbers
#     'City_Encoded': [0, 1, 2, 0, 1]            # Multi-class encoded (0, 1, 2)
# }

# df_test = pd.DataFrame(data)

# print("--- Original Data ---")
# print(df_test)
# print(f"\nOriginal Means:\n{df_test.mean()}")

# # 2. Run the scaling function
# df_scaled = scale_features(df_test)

# print("\n--- Scaled Data ---")
# print(df_scaled)

# # 3. Verification of Standardization
# print("\n--- Verification ---")
# print(f"New Means (should be approx 0):")
# print(df_scaled.mean().round(2))
# print(f"\nNew Std Dev (should be 1):")
# print(df_scaled.std().round(2))

###################################removing irrelevant features##########################################

# # 1. Create a dataset with "Noise"
# data = {
#     'Student_ID': [101, 102, 103, 104],       # Should be dropped (Keyword 'ID')
#     'Full_Name': ['Ali', 'Sara', 'Zain', 'Abu'], # Should be dropped (Keyword 'Name')
#     'Age': [20, 22, 21, 23],                  # Should be KEPT (Useful)
#     'Country': ['Pakistan', 'Pakistan', 'Pakistan', 'Pakistan'], # Should be dropped (Zero Variance)
#     'Email_Address': ['a@x.com', 's@x.com', 'z@x.com', 'ab@x.com'], # Should be dropped (100% Unique text)
#     'Score': [85, 90, 78, 92]                 # Should be KEPT (Useful)
# }

# df_test = pd.DataFrame(data)

# print("--- Original Columns ---")
# print(df_test.columns.tolist())

# # 2. Run the function
# df_cleaned = drop_irrelevant_columns(df_test)

# print("\n--- Columns After Automated Dropping ---")
# print(df_cleaned.columns.tolist())

# # 3. Validation Logic
# expected_keep = ['Age', 'Score']
# actual_keep = df_cleaned.columns.tolist()

# if set(actual_keep) == set(expected_keep):
#     print("\nSUCCESS: Only relevant features ('Age' and 'Score') remain.")
# else:
#     print("\nCHECK: Some irrelevant columns might still be present.")


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>EDA TESTS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# data = {'Age': [20, 25], 'City': ['Karachi', 'Lahore']}
# df = pd.DataFrame(data)

# get_basic_info(df)

########################################### get_statistical_summary(df) ############################

# from eda import get_statistical_summary

# # 1. Create a dataset with specific numeric patterns
# data = {
#     'Age': [20, 21, 19, 22, 50],           # One high value (50) will pull the Mean up
#     'Salary': [50000, 52000, 48000, 51000, 49000], # Tight range (Low Std Dev)
#     'Scores': [0, 100, 50, 75, 25],        # Wide range (High Std Dev)
#     'Department': ['HR', 'IT', 'IT', 'HR', 'IT']   # Text column (Should be ignored by function)
# }

# df_test = pd.DataFrame(data)

# # 2. Run the test
# print("--- TEST DATA ---")
# print(df_test)

# get_statistical_summary(df_test)


#################################missing value analysis##########################################

# from eda import get_missing_values_analysis

# # Create data with some NaN (missing) values
# data = {
#     'A': [1, 2, np.nan, 4],
#     'B': [np.nan, np.nan, np.nan, 4],
#     'C': [1, 2, 3, 4]
# }
# df_test = pd.DataFrame(data)

# get_missing_values_analysis(df_test)

#########################################unique value analysis##########################################

# from eda import get_unique_values_analysis

# data = {
#     'Gender': ['M', 'F', 'M', 'F'],          # Binary (2)
#     'Country': ['Pakistan', 'Pakistan', 'Pakistan', 'Pakistan'], # Constant (1)
#     'User_ID': [101, 102, 103, 104],        # Unique ID (4)
#     'Rating': [1, 2, 3, 1]                   # Categorical (3)
# }
# df_test = pd.DataFrame(data)

# get_unique_values_analysis(df_test)

##################################correlation analysis##########################################

# from eda import get_correlation_analysis

# data = {
#     'Hours_Studied': [1, 2, 3, 4, 5],
#     'Test_Score': [50, 60, 70, 80, 90], # Perfectly correlated with hours
#     'Random_Noise': [10, 2, 85, 4, 12],  # No correlation
#     'Sleep_Hours': [8, 7, 6, 5, 4]      # Negative correlation
# }
# df_test = pd.DataFrame(data)

# get_correlation_analysis(df_test)

########################################distribution analysis##########################################


# from eda import get_distribution_analysis

# data = {
#     'Normal_Data': [10, 11, 10, 12, 11, 10, 9, 11], 
#     'Right_Skew':  [1, 2, 2, 3, 4, 10, 50, 100],    
#     'Left_Skew':   [1, 50, 90, 95, 98, 99, 100, 101] # Added one value (101) to make it 8
# }
# df_test = pd.DataFrame(data)

# get_distribution_analysis(df_test)

###############################outlier analysis##########################################


# from eda import get_outlier_analysis

# # 1. Create data with intentional outliers
# data = {
#     'Salary': [25000, 27000, 26000, 28000, 24000, 25500, 150000], # 150k is the outlier
#     'Age': [22, 23, 21, 24, 22, 23, 85],                         # 85 is the outlier
#     'Fixed_Value': [10, 10, 10, 10, 10, 10, 10]                  # No outliers here
# }

# df_test = pd.DataFrame(data)

# # 2. Run the test
# get_outlier_analysis(df_test)

######################################get_target_analysis##########################################



# from eda import get_target_analysis

# 1. Create a dummy dataset
# data = {
#     'Input_1': [1, 2, 3, 4, 5],
#     'Price_Target': [250.50, 310.00, 450.75, 120.25, 600.00] # Continuous values
# }

# data = {
#     'Feature_A': [10, 20, 30, 40, 50],
#     'Feature_B': [1.1, 2.2, 3.3, 4.4, 5.5],
#     'Target_Class': ['Yes', 'No', 'Yes', 'Yes', 'No']
# }

# df_test = pd.DataFrame(data)

# # 2. Set the target column name statically as requested
# static_target = 'Target_Class' 

# # 3. Call the function
# get_target_analysis(df_test, static_target)


#################################get_automated_insights##########################################


# from eda import get_automated_insights

# # 1. Create a dataset with specific "problems" to trigger insights
# data = {
#     'Feature_A': [10, 12, 11, 13,14],        # 100 is an outlier
#     'Feature_B': [1, 2, 3, 4, 5],         # np.nan is a missing value
#     'Feature_C': [1, 2, 3, 5,6]             # High skewness
# }
# df_test = pd.DataFrame(data)

# # 2. Run the test
# get_automated_insights(df_test)

########################################full eda report##########################################

# from eda import run_full_eda

# # Create a messy dataset to see the engine in action
# data = {
#         # 1. Outliers & Missing Values
#         'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 200, np.nan, 210],
        
#         # 2. Highly Skewed Data (exponentially increasing)
#         'Income_Growth': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000],
        
#         # 3. Categorical Data with Duplicates
#         'Department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales'],
        
#         # 4. Target Column (Imbalanced)
#         'Approved': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No']
#     }
# df_test = pd.DataFrame(data)

# # Run everything with one command
# run_full_eda(df_test, target_col='Approved')
####################################Train Modules Tests##########################################



###################################### detect_problem_type ##########################
# from train_modules import split_data

# data =  {
#         # 1. Outliers & Missing Values
#         'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 200, np.nan, 210],
        
#         # 2. Highly Skewed Data (exponentially increasing)
#         'Income_Growth': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000],
        
#         # 3. Categorical Data with Duplicates
#         'Department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'IT', 'Sales'],
        
#         # 4. Target Column (Imbalanced)
#         'Approved': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'No', 'No']
#     }
# df_test = pd.DataFrame(data)
# # Classification Problem
# result = split_data(df_test, 'Income_Growth')

##############################train_classification_models##########################


# from train_modules import split_data, train_classification_models

# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Income': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Approved': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
# }

# df_test = pd.DataFrame(data)

# # Step 1: Split data
# split_result = split_data(df_test, 'Approved')

# # Step 2: Get X_train and y_train
# X_train = split_result['X_train']
# y_train = split_result['y_train']

# # Step 3: Train models
# models_result = train_classification_models(X_train, y_train)


###############################train_regression_models##########################

# from train_modules import split_data, train_regression_models


# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Education_Level': [1, 2, 3, 4, 2, 3, 2, 4, 4, 1, 1, 2, 3, 3, 4, 4, 4, 2, 2, 4],
#     'Salary': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000]
# }

# df_test = pd.DataFrame(data)

# split_result = split_data(df_test, 'Salary')

# X_train = split_result['X_train']
# y_train = split_result['y_train']

# models_result = train_regression_models(X_train, y_train)

# print(models_result)

#############################evaluate_classification_models########################## 

# from train_modules import split_data, train_classification_models, evaluate_classification_models
# import pandas as pd

# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Income': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Approved': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
# }

# df_test = pd.DataFrame(data)

# split_result = split_data(df_test, 'Approved')

# X_train = split_result['X_train']
# X_test = split_result['X_test']
# y_train = split_result['y_train']
# y_test = split_result['y_test']

# models_result = train_classification_models(X_train, y_train)
# models = models_result['models']

# evaluation_result = evaluate_classification_models(models, X_test, y_test)

# print(evaluation_result)

################################evaluate_regression_models##########################

# from train_modules import split_data, train_regression_models, evaluate_regression_models


# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Education_Level': [1, 2, 3, 4, 2, 3, 2, 4, 4, 1, 1, 2, 3, 3, 4, 4, 4, 2, 2, 4],
#     'Salary': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000]
# }

# df_test = pd.DataFrame(data)

# split_result = split_data(df_test, 'Salary')

# X_train = split_result['X_train']
# X_test = split_result['X_test']
# y_train = split_result['y_train']
# y_test = split_result['y_test']

# models_result = train_regression_models(X_train, y_train)
# models = models_result['models']

# evaluation_result = evaluate_regression_models(models, X_test, y_test)

# print(evaluation_result)

##################################automl tests##########################################

####################################select_best_model#######################

# from train_modules import split_data, train_regression_models, evaluate_regression_models
# from autoML import select_best_model

# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Education_Level': [1, 2, 3, 4, 2, 3, 2, 4, 4, 1, 1, 2, 3, 3, 4, 4, 4, 2, 2, 4],
#     'Salary': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000]
# }

# df_test = pd.DataFrame(data)

# split_result = split_data(df_test, 'Salary')

# X_train = split_result['X_train']
# X_test = split_result['X_test']
# y_train = split_result['y_train']
# y_test = split_result['y_test']

# models_result = train_regression_models(X_train, y_train)
# models = models_result['models']

# evaluation_result = evaluate_regression_models(models, X_test, y_test)
# evaluation = evaluation_result['evaluation']

# best_model_result = select_best_model(evaluation, 'regression')

# print(best_model_result)


#################################auto_feature_selection##########################

# from autoML import auto_feature_selection


# data_regression = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Education_Level': [1, 2, 3, 4, 2, 3, 2, 4, 4, 1, 1, 2, 3, 3, 4, 4, 4, 2, 2, 4],
#     'Random_Col': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#     'Salary': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000]
# }

# df_test = pd.DataFrame(data_regression)
# X = df_test.drop('Salary', axis=1)
# y = df_test['Salary']

# result = auto_feature_selection(X, y, problem_type='regression')

############################auto_hyperparameter_tuning##############################################


# from train_modules import split_data, train_classification_models, evaluate_classification_models
# from autoML import  select_best_model, auto_hyperparameter_tuning
# import pandas as pd

# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Income': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Approved': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
# }

# df_test = pd.DataFrame(data)

# split_result = split_data(df_test, 'Approved')

# X_train = split_result['X_train']
# X_test = split_result['X_test']
# y_train = split_result['y_train']
# y_test = split_result['y_test']

# models_result = train_classification_models(X_train, y_train)
# models = models_result['models']

# evaluation_result = evaluate_classification_models(models, X_test, y_test)
# evaluation = evaluation_result['evaluation']

# best_model_result = select_best_model(evaluation, 'classification')
# best_model_name = best_model_result['best_model']

# tuning_result = auto_hyperparameter_tuning(best_model_name, X_train, y_train, 'classification')

# print(tuning_result)

# from autoML import generate_model_summary

# all_models_ranked = [
#     {'model': 'Logistic Regression', 'score': 0.94},
#     {'model': 'Random Forest', 'score': 0.88},
#     {'model': 'Decision Tree', 'score': 0.82}
# ]

# result = generate_model_summary(
#     best_model_name='Logistic Regression',
#     best_score=0.94,
#     all_models_ranked=all_models_ranked,
#     problem_type='classification'
# )

# print(result)



########################autopipeline##########################

# from train_modules import split_data, train_classification_models, evaluate_classification_models
# from autoML import run_automl_pipeline
# import pandas as pd

# data = {
#     'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
#     'Income': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000],
#     'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
#     'Approved': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
# }

# df_test = pd.DataFrame(data)

# split_result = split_data(df_test, 'Approved')

# X_train = split_result['X_train']
# y_train = split_result['y_train']

# models_result = train_classification_models(X_train, y_train)
# models = models_result['models']

# evaluation_result = evaluate_classification_models(models, X_train, y_train)
# evaluation = evaluation_result['evaluation']

# pipeline_result = run_automl_pipeline(models, evaluation, X_train, y_train, 'classification')

# print(pipeline_result)


#################################test full pipeline##########################################
import pandas as pd
import numpy as np
from loader import load_file
from data_cleaner import (handle_missing_values, remove_duplicates, 
                                   format_correction, fix_data_types, 
                                   handle_outliers, drop_irrelevant_columns)
from eda import run_full_eda
from train_modules import (detect_problem_type, split_data, 
                                    train_classification_models, train_regression_models,
                                    evaluate_classification_models, evaluate_regression_models)
from autoML import (select_best_model, auto_feature_selection, 
                            auto_hyperparameter_tuning, generate_model_summary, 
                            run_automl_pipeline)
from prediction import predict_new_data
from generate_report import generate_report

print("="*70)
print("         FULL PROJECT PIPELINE TEST")
print("="*70)

data = {
    'Age': [25, 28, 30, 32, 29, 31, 27, 33, 35, 22, 24, 26, 28, 30, 29, 31, 32, 26, 29, 31],
    'Income': [30000, 35000, 40000, 45000, 38000, 42000, 32000, 48000, 50000, 25000, 28000, 36000, 39000, 43000, 41000, 46000, 49000, 34000, 37000, 44000],
    'Experience': [1, 2, 3, 4, 2, 3, 2, 4, 5, 1, 1, 2, 3, 3, 4, 4, 5, 2, 2, 4],
    'Approved': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

print("\n✅ STEP 1: DATA LOADING")
print(f"Data loaded: {df.shape}")

print("\n✅ STEP 2: DATA CLEANING")
df = handle_missing_values(df)
df = remove_duplicates(df)
df = format_correction(df)
df = fix_data_types(df)
df = handle_outliers(df)
df = drop_irrelevant_columns(df)
print(f"Data cleaned: {df.shape}")

print("\n✅ STEP 3: EDA")
eda_results = run_full_eda(df, target_col='Approved')

print("\n✅ STEP 4: DETECT PROBLEM TYPE")
problem_detection = detect_problem_type(df, 'Approved')
problem_type = problem_detection['problem_type']
print(f"Problem Type: {problem_type}")

print("\n✅ STEP 5: DATA SPLITTING")
split_result = split_data(df, 'Approved', problem_type)
X_train = split_result['X_train']
X_test = split_result['X_test']
y_train = split_result['y_train']
y_test = split_result['y_test']

print("\n✅ STEP 6: MODEL TRAINING")
if problem_type == 'classification':
    models_result = train_classification_models(X_train, y_train)
else:
    models_result = train_regression_models(X_train, y_train)
models = models_result['models']
print(f"Models trained: {len(models)}")

print("\n✅ STEP 7: MODEL EVALUATION")
if problem_type == 'classification':
    evaluation_result = evaluate_classification_models(models, X_test, y_test)
else:
    evaluation_result = evaluate_regression_models(models, X_test, y_test)
evaluation = evaluation_result['evaluation']

print("\n✅ STEP 8: AUTOML PIPELINE")
pipeline_result = run_automl_pipeline(models, evaluation, X_train, y_train, problem_type)
best_model = pipeline_result['best_model']
best_model_name = pipeline_result['best_model_name']
best_score = pipeline_result['best_score']
best_parameters = pipeline_result['best_parameters']
selected_features = pipeline_result['selected_features']
removed_features = pipeline_result['removed_features']

print("\n✅ STEP 9: PREDICTIONS ON NEW DATA")

# Use ONLY selected features for prediction
X_test_selected = X_test[selected_features]

prediction_result = predict_new_data(best_model, X_test_selected)

if prediction_result['success']:
    predictions = prediction_result['predictions']
    print(f"Predictions made: {len(predictions)}")
    print(f"Sample predictions: {predictions[:5]}")
else:
    print(f"Prediction failed: {prediction_result['error']}")

print("\n✅ STEP 10: GENERATE REPORT")
report_result = generate_report(
    problem_type=problem_type,
    evaluation=evaluation,
    best_model=best_model_name,
    best_score=best_score,
    best_params=best_parameters,
    selected_features=selected_features,
    removed_features=removed_features,
    dataset_shape=df.shape
)

print("\n" + "="*70)
print("         ALL TESTS COMPLETED SUCCESSFULLY ✅")
print("="*70)