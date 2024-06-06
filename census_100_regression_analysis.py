import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

scaler = StandardScaler()
models = {
    'OLS': Pipeline([('scaler', scaler), ('regressor', LinearRegression())]),
    'Ridge': Pipeline([('scaler', scaler), ('regressor', RidgeCV(alphas=np.logspace(-6, 6, 13)))]),
    'Lasso': Pipeline([('scaler', scaler), ('regressor', LassoCV(alphas=np.logspace(-6, 6, 13), max_iter=10000, cv=5))]),
    'Elastic Net': Pipeline([('scaler', scaler), ('regressor', ElasticNetCV(alphas=np.logspace(-6, 6, 13), l1_ratio=[.1, .5, .7, .9, .95, 1], cv=5))]),
    'SVM': Pipeline([('scaler', scaler), ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))]),
    'Random Forest': Pipeline([('regressor', RandomForestRegressor(n_estimators=100, random_state=42))]),
    'Gradient Boosting': Pipeline([('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])}


class regression_analysis:
    def __init__(self, grid_100_qual, grid_100_features) -> None:
            self.grid_100_qual = grid_100_qual
            self.grid_100_features = grid_100_features
            self.grid_100_prediction = None


    def clean_outliers(self, df, col="2020_heat", factor=1, lower=True, upper=True):

        # Calculate the first and third quartiles (Q1 and Q3)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1

        # set default values
        lower_bound = df[col].min()
        upper_bound = df[col].max()
        
        print(f"FACTOR: {factor}")
        # Define the lower and upper bounds
        if lower:
            lower_bound = Q1 - factor * IQR
            print(f"lower bound is {int(lower_bound)}. culc as {Q1} - {round(factor, 2)} * {int(IQR)}")
        if upper:
            upper_bound = Q3 + factor * IQR
            print(f"upper bound is {int(upper_bound)}. culc as {Q3} - {round(factor, 2)} * {int(IQR)}")

        # Filter out values outside the lower and upper bounds
        return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    def get_X_y(self, factor=1, threshold=1, col="2020_heat"):
        grid_100_thresh = self.grid_100_qual[["rel_ratio", "2020_heat"]].copy()
        grid_100m = self.grid_100_features.copy()

        # Filter the dataset based on the current threshold
        grid_100m = grid_100m.merge(grid_100_thresh, left_index=True, right_index=True, how='inner')
        
        if threshold:
            grid_100m = grid_100m[grid_100m["rel_ratio"] >= threshold]

        if factor:
            grid_100m = self.clean_outliers(grid_100m.fillna(0).copy(), col=col, factor=factor, lower=True, upper=True)
        
        # Prepare features and target variable
        X = grid_100m.drop(['2020_heat', 'rel_ratio'], axis=1)
        y = grid_100m['2020_heat']
        return X, y

    
    def seven_regressions(self, X, y):
    
        for name, model in models.items():
            # Cross-validate model for MSE
            mse_scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
            # Cross-validate model for R²
            r2_scores = cross_val_score(model, X, y, cv=10, scoring='r2')

            print(f"{name} Regression:")
            print(f"Mean Squared Error: {-np.mean(mse_scores):.4f} (+/- {np.std(mse_scores):.4f})")
            print(f"R² Score: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})\n")
            
            
    def best_threshold(self, thresholds = np.linspace(0.8, 1, num=11), factor=1, model_selection = ['Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting']):
        scores_dict = {model_name: [] for model_name in model_selection}
        high_scores_dict = {model_name: 0 for model_name in model_selection}
        high_thresholds_dict = {model_name: 0 for model_name in model_selection}

        for threshold in thresholds:

            X, y = self.get_X_y(factor=factor, threshold=threshold)          

            for model_name, model in models.items():
                if model_name in model_selection:

                    # Split the data into training and test sets
                    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train the model and evaluate using cross-validation
                    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    scores_dict[model_name].append(np.mean(scores))

                    if high_scores_dict[model_name] <= np.mean(scores):
                        high_scores_dict[model_name] = np.mean(scores)
                        high_thresholds_dict[model_name] = threshold

        # Plotting the results
        plt.figure(figsize=(10, 6))
        for model_name, scores in scores_dict.items():
            plt.plot(thresholds, scores, marker='o', label=model_name)

        plt.title('Model Performance vs. Data Quality Threshold')
        plt.xlabel('Minimum rel_ratio Threshold')
        plt.ylabel('Cross-Validated R² Score')
        plt.legend()
        plt.grid(True)
        plt.show()

        return high_thresholds_dict

    def pred_vs_actual(self, factor=1, threshold=1, model='Ridge', random_state=42):
        
        X, y = self.get_X_y(factor=factor, threshold=threshold)  
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


        # Create and fit a Ridge Regression model
        selected_model = models[model]  # You can adjust the regularization strength (alpha) as needed
        selected_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = selected_model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Squared Error:", mse)
        r2 = r2_score(y_test, y_pred)
        print("R² score: ", r2)
        # Plotting the predicted values against the actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='r', label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Prediction vs Actual Values \n Model: {model} | cleaned | c_abs_urban_rs\n treshholde = {threshold} | factor = {factor} \n R²={round(r2, 2)} | mse={abs(mse.astype(int))} | mae={mae.astype(int)}')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
    def best_IQR_clean(self, threshold=1, factors = np.arange(1.1, 0.8, -0.01), model_selection=['Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting']):

        # Initialize a list to store R² scores and corresponding factors
        r2_scores =  {model_name: [] for model_name in model_selection}
        high_scores_dict = {model_name: 0 for model_name in model_selection}
        high_IQR_clean_dict = {model_name: 0 for model_name in model_selection}

        for factor in factors:
            X, y = self.get_X_y(factor=factor, threshold=threshold)  

            for model_name, model in models.items():
                    if model_name in model_selection:
                        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                        r2_scores[model_name].append(np.mean(scores))

                        if high_scores_dict[model_name] <= np.mean(scores):
                                high_scores_dict[model_name] = np.mean(scores)
                                high_IQR_clean_dict[model_name] = factor

        '''means = [sum(sub_array) / len(sub_array) for sub_array in r2_scores]
        # Plotting the R² scores against the factors
        plt.figure(figsize=(10, 6))
        plt.plot(factors, means, marker='o')
        plt.xlabel('Outlier Cleaning Factor')
        plt.ylabel('R² Score')
        plt.title(f'R² Score by Outlier Cleaning Factor \n Model: c_abs_urban_rs | threshold: {threshold}')
        plt.grid(True)
        plt.show()'''

        
         # Plotting the results
        plt.figure(figsize=(10, 6))
        for model_name in model_selection:
            plt.plot(factors, r2_scores[model_name], marker='o', label=model_name)

        plt.title('Model Performance vs. IQR Cleaning Factor')
        plt.xlabel('IQR Cleaning Factor')
        plt.ylabel('Cross-Validated R² Score')
        plt.legend()
        plt.grid(True)
        plt.show()

        return high_IQR_clean_dict
    
    def ridge_lasso_coefficients(self, factor=1, threshold=1,random_state=42):
        X, y = self.get_X_y(factor=factor, threshold=threshold)
        # Generate synthetic data
        feature_names = X.columns

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Lasso regression with cross-validation
        lasso = LassoCV(alphas=np.logspace(-6, 6, 26), max_iter=100000, cv=5, random_state=42)
        lasso.fit(X_scaled, y)

        ridge = RidgeCV(alphas=np.logspace(-6, 6, 26), cv=5)
        ridge.fit(X_scaled, y)

        # Coefficients
        coef_lasso = pd.Series(lasso.coef_, index=feature_names).sort_values(key=abs, ascending=False)
        coef_ridge = pd.Series(ridge.coef_, index=feature_names).sort_values(key=abs, ascending=False)
        
        print(f"Alphas: Lasso: {lasso.alpha_}; Ridge: {ridge.alpha_}")
        print(f"intercepts: Lasso: {lasso.intercept_}; Ridge: {ridge.intercept_}")
        
        scores_lasso = cross_val_score(lasso, X_scaled, y, cv=5, scoring='r2')
        scores_ridge_r2 = cross_val_score(ridge, X_scaled, y, cv=10, scoring='r2')
        scores_ridge_nmse = cross_val_score(ridge, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
        print("Zensus + Remote Sensing + AHK+ ALKIS btypology")
        print(f"threshold: {threshold}; factor: {factor}")
        print(f"Alphas: Lasso: {lasso.alpha_}; Ridge: {ridge.alpha_}")
        print(f"intercepts: Lasso: {lasso.intercept_}; Ridge: {ridge.intercept_}")
        # print("Cross-validated R² scores:", scores_lasso)
        print(f"Average R²: Lasso: {np.mean(scores_lasso):.3f} (+/- {np.std(scores_lasso):.2f}); Ridge: {np.mean(scores_ridge_r2):.3f} (+/- {np.std(scores_ridge_r2):.2f})")
        return coef_ridge, coef_lasso
    
    def pred(self, factor=1, threshold=1, model='Ridge', random_state=42):
        
        X, y = self.get_X_y(factor=factor, threshold=threshold)  

        # Create and fit a Ridge Regression model
        selected_model = models[model]  # You can adjust the regularization strength (alpha) as needed
        selected_model.fit(X, y)
        
        # tiles to predict values for
        tiles = self.grid_100_features.dropna()

        # Make predictions on the test set
        y_pred = selected_model.predict(tiles)
        y_pred = pd.Series(data=y_pred, index=tiles.index)
        y_pred = y_pred.rename('prediction')
        self.grid_100_prediction = y_pred
        return self.grid_100_prediction.copy()
    
    def get_results(self):
        return pd.concat([self.grid_100_features, self.grid_100_qual[['2020_heat', 'geometry']], self.grid_100_prediction], axis=1, join = 'inner').fillna(0)
        