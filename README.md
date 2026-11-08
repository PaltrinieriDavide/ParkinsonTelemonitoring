# Parkinsons Telemonitoring

**Report for the Foundations of Machine Learning Exam**

**Davide Paltrinieri**  
Date: June 2024  

## Abstract

This document describes the process and design choices involved in developing the project for the Foundations of Machine Learning exam on 17/07/2024.

## 1 Introduction to the problem

The Parkinsons Telemonitoring dataset contains biomedical data on the early stages of Parkinson’s disease. Specifically, the dataset contains information on 42 patients, with approximately 200 samples per patient and a total of 5875 samples. The main objective is to predict the target variables motor UPDRS and total UPDRS, which refer to the UPDRS (Unified Parkinson’s Disease Rating Scale), a scale commonly used to monitor the progression of Parkinson’s disease.

## 2 Project structure

The project is organized as follows:

- `train.py`
- The `pt_dataset` folder, which contains the dataset (in `.csv` format as well as in `.data` and `.names` formats)
- `data.log`, a log file containing all information about the models, program execution, and the results obtained at each step. Each run clears the file and rewrites it with the new information. If the file is missing, the program creates it.
- The `graphs` folder, which contains the plots generated at the end of the run (shown in Section 5).

To get started, run `train.py`, optionally passing the dataset as an argument (if no path is provided, `pt_dataset\parkinsons_updrs.data` is used by default).

## 3 Data analysis and management

### 3.1 EDA

As described above, the dataset contains 5875 samples, each with 20 features. These are primarily integers (int64) and real numbers (float64), with one categorical variable. The features are briefly described below:

- `subject number`, an integer between 1 and 42 that identifies the subject (int64)
- `subject age`
- `subject gender`, the only categorical feature
- `test time`, the time spent in the study
- 16 biomedical measures (all float64)

The dataset is complete, with no missing values in any sample. The dataset was split into training (70%) and testing (30%) sets.

### 3.2 Data scaling

An analysis of the dataset showed that the feature values span different ranges, so I chose to scale them using **Standardization**.

### 3.3 Feature selection

Given the number of features relative to the number of samples in the dataset, feature selection did not seem strictly necessary. I chose to apply it anyway, which improved the final results. Specifically, because we tested different models, I decided to use a wrapper method to tailor feature selection to each model. I also decided to remove the `subject number` feature from the dataset because I considered it particularly ineffective for prediction. Each model described in Section 4.2 was therefore tested and evaluated both with and without feature selection. Section 4.3 reports the results.

## 4 Cross-validation and model testing

This section presents the models tested, the metrics used, and the corresponding cross-validation results.

### 4.1 Metrics

The following metrics were used to evaluate the models:

- `neg mean absolute error`
- `neg mean squared error`
- `neg root mean squared error`
- `r2`

### 4.2 Models and hyperparameters

Several models were tested, each twice, once with feature selection and once without it. Two pipelines were created for each model. The first contained the regressor and a Sequential Feature Selector. The second contained only the model. A grid search was then run on each pipeline to find the combination of hyperparameters that offered the best generalization. The models and the parameter values tested are shown below:

- **Linear Regression**
- **K-Nearest Neighbors**  
  - K = [3, 5, 7, 9]
- **Support Vector Regressor**  
  - C = [1, 10]  
  - gamma = [0.001, 0.01]
- **Decision Tree**  
  - max depth = [10, 20, None]  
  - min samples split = [2, 5]
- **Random Forest**  
  - number of estimators = [50, 100, 200]  
  - min samples split = [2, 5, 10]

### 4.3 Results

#### 4.3.1 motor UPDRS

After the steps described in the previous section (4.2), cross-validation was used to select the optimal model.  
Note: The following tables refer to the motor UPDRS target variable.

**Table 1: motor UPDRS with feature selection**

| Model                 | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -6.4127   | -57.3823  | -7.5722   | 0.1370  |
| KNeighborsRegressor   | -3.1961   | -21.9084  | -4.6771   | 0.6698  |
| SVR                   | -4.8802   | -39.2567  | -6.2644   | 0.4090  |
| DecisionTreeRegressor | -0.5196   | -3.4268   | -1.8450   | 0.9485  |
| RandomForestRegressor | -0.6011   | -1.9606   | -1.3972   | 0.9705  |

**Table 2: motor UPDRS without feature selection**

| Model                 | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -6.3639   | -56.8181  | -7.5341   | 0.1456  |
| KNeighborsRegressor   | -3.1725   | -24.2967  | -4.9276   | 0.6343  |
| SVR                   | -5.3555   | -46.3815  | -6.8088   | 0.3020  |
| DecisionTreeRegressor | -0.7925   | -6.4843   | -2.5272   | 0.9029  |
| RandomForestRegressor | -0.8700   | -2.8121   | -1.6753   | 0.9576  |

#### 4.3.2 total UPDRS

**Table 3: total UPDRS with feature selection**

| Model                 | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -8.1429   | -97.1317  | -9.8520   | 0.1580  |
| KNeighborsRegressor   | -4.1199   | -37.5284  | -6.1237   | 0.6737  |
| SVR                   | -6.5175   | -77.0747  | -8.7762   | 0.3319  |
| DecisionTreeRegressor | -0.6273   | -5.2887   | -2.2959   | 0.9541  |
| RandomForestRegressor | -0.8147   | -3.5844   | -1.8894   | 0.9689  |

**Table 4: total UPDRS without feature selection**

| Model                 | neg MAE   | neg MSE   | neg RMSE  | R2      |
|-----------------------|-----------|-----------|-----------|---------|
| LinearRegression      | -8.1095   | -96.3776  | -9.8134   | 0.1645  |
| KNeighborsRegressor   | -4.2663   | -44.9509  | -6.7042   | 0.6098  |
| SVR                   | -7.0332   | -87.5793  | -9.3551   | 0.2409  |
| DecisionTreeRegressor | -0.9981   | -10.3635  | -3.1671   | 0.9093  |
| RandomForestRegressor | -1.0690   | -4.5575   | -2.1306   | 0.9604  |

## 5 Conclusions

This section draws conclusions from the information and data presented above. The metrics in the tables show a marked improvement in results when using models such as the Decision Tree and Random Forset. The next two plots illustrate this more clearly. They show a set of points for motor UPDRS and total UPDRS, respectively, where the x-coordinate is the predicted target value on the test set and the y-coordinate is the actual target value on the test set. A model is therefore more effective when its points lie closer to a straight line at a 45° angle.

It is worth noting that Random Forset with feature selection was the optimal model for both target variables and performed well on the test set. The test set results and selected features are shown below:

**Model motor UPDRS**  
neg MAE -0.493563360181508  
neg MSE -1.306831472688216  
neg RMSE -1.1431672986436483  
R2 0.9799484245181771  

**Table 5: motor UPDRS final results**

**Model total UPDRS**  
neg MAE -0.566947604651162  
neg MSE -1.8880643708320002  
neg RMSE -1.3740685466278602  
R2 0.9832142624649486  

## Plots and Visualizations

![Motor UPDRS Plot](motor_UPDRS_RandomForestRegressor_FeatureSelection.png)

![Total UPDRS Plot](total_UPDRS_RandomForestRegressor_FeatureSelection.png)


**Table 6: total UPDRS final results**

The selected features are:

- motor UPDRS: age, sex, test time, Jitter(%), Jitter(Abs), Jitter:PPQ5, NHR, RPDE, DFA
- total UPDRS: age, sex, test time, Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP, DFA

## References

Athanasios Tsanas and Max Little. Parkinsons Telemonitoring. UCI Machine Learning Repository, 2009. DOI: [https://doi.org/10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N).
