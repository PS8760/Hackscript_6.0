# Development of AI-ML based models for predicting prices of agri-horticultural commodities

## Description
The Department of Consumer Affairs monitors the daily prices of 22 essential food commodities through 550 price reporting centres across the country. The Department also maintains buffer stock of pulses, viz., gram, tur, urad, moon and masur, and onion for strategic market interventions to stabilize the volatility in prices. Decisions for market interventions such as release of stocks from the buffer are taken on the basis of the price trends and outlook. At present, the analyses of prices are based on the seasonality, historical and emerging trends, market intelligence inputs, crop sowing and production estimates. ARIMA based economic models have also been used to examine and forecast prices of pulses.

## Approach Towards the Solution

### Seasonality
- Prices often follow seasonal patterns (e.g., onion prices spike during monsoon due to supply shortages).

### Historical Trends
- Past price data is used to predict future trends.

### Market Intelligence
- Reports on demand-supply gaps, trade dynamics, and other market factors.

### Crop Sowing and Production Estimates
- Data on crop yields and sowing patterns.

### ARIMA Models
- Traditional time-series models for price forecasting.

However, these methods may not fully capture the complexity of price dynamics, especially with external factors like weather, global market trends, and supply chain disruptions.

## Step 1: Define the Scope
- **Focus on key commodities:** Start with pulses (gram, tur, urad, moon, masur) and onion, as they are critical for buffer stock interventions.
- **Objective:** Build a model to predict prices 1-2 weeks in advance to enable timely market interventions.

## Step 2: Data Collection and Preprocessing
### Data Sources:
- **Historical Price Data:** Daily prices from 550 reporting centers.
- **Buffer Stock Data:** Information on buffer stock levels of pulses and onion.
- **Crop Data:** Sowing and production estimates from government reports.
- **Weather Data:** Rainfall, temperature, and humidity data from meteorological departments.
- **Market Intelligence:** Reports on demand-supply gaps, trade policies, and global market trends.
- **External Factors:** Fuel prices, transportation costs, and inflation rates.

### Data Preprocessing:
- Clean data (handle missing values, outliers).
- Normalize/standardize data for model training.
- Create time-series features (e.g., lagged prices, moving averages).
- Integrate external factors (e.g., weather indices, global price trends).

## Step 3: Feature Engineering
- **Temporal Features:** Lagged prices, moving averages, seasonality indicators.
- **Weather Features:** Rainfall, temperature, humidity indices.
- **Crop Features:** Sowing area, production estimates, yield trends.
- **Market Features:** Buffer stock levels, demand-supply gaps, global prices.
- **Economic Features:** Inflation rates, fuel prices, transportation costs.

## Step 4: Model Selection
Use a hybrid approach combining traditional and advanced models:

### Baseline Models:
- **ARIMA/SARIMA:** Traditional time-series models for benchmarking.
- **Prophet:** Facebook’s time-series model for seasonality and trend analysis.

### Machine Learning Models:
- **Random Forest:** For handling non-linear relationships.
- **XGBoost/LightGBM:** Gradient boosting models for high accuracy.

### Deep Learning Models:
- **LSTM/GRU:** Recurrent neural networks for capturing temporal dependencies.

### Hybrid Model:
- Combine ARIMA (for linear trends) with LSTM (for non-linear patterns).

## Step 5: Model Training and Validation
### Train-Test Split:
- Use 70-80% of data for training and 20-30% for testing.
- Ensure the split respects temporal order (no data leakage).

### Cross-Validation:
- Use time-series cross-validation to evaluate model performance.

### Evaluation Metrics:
- **RMSE (Root Mean Squared Error):** Measures prediction accuracy.
- **MAE (Mean Absolute Error):** Average error magnitude.
- **MAPE (Mean Absolute Percentage Error):** Percentage error for interpretability.

## Step 6: Model Deployment
### Real-Time Predictions:
- Deploy the best-performing model on a cloud platform (AWS, Google Cloud, Azure).
- Use APIs to provide real-time price predictions.

### Dashboard:
- Build a user-friendly dashboard (using Tableau, Power BI, or Streamlit) to visualize price trends and predictions.
- Include features like:
  - Historical price trends.
  - Predicted prices for the next 1-2 weeks.
  - Key drivers of price changes (e.g., weather, buffer stock levels).

## Step 7: Monitoring and Maintenance
- Continuously monitor model performance.
- Retrain the model with new data to adapt to changing market dynamics.
- Incorporate feedback from policymakers to improve the system.

## Innovative Features to Stand Out
### Explainable AI:
- Use SHAP (SHapley Additive exPlanations) or LIME to explain model predictions.
- Help policymakers understand the key drivers of price changes.

### Scenario Analysis:
- Simulate the impact of different market interventions (e.g., releasing buffer stocks, import/export policies).

### Early Warning System:
- Build an alert system to notify policymakers when prices are predicted to cross critical thresholds.

### Scalability:
- Design the system to be easily adaptable to other commodities and regions.

## Tech Stack
- **Programming Languages:** Python, R.
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch, Statsmodels, Prophet.
- **Visualization:** Matplotlib, Seaborn, Tableau, Streamlit.
- **Cloud Platforms:** AWS, Google Cloud, or Azure for deployment.

---

## ARIMA
### Why ARIMA is a Good Choice
- **Interpretability:** ARIMA is a statistical model, and its parameters (p, d, q) have clear meanings.
- **Handles Seasonality and Trends:** ARIMA can model seasonality and trends in price data.
- **Low Computational Cost:** ARIMA is computationally lightweight compared to deep learning models.
- **Proven Track Record:** ARIMA has been used in many real-world forecasting problems.

### Step-by-Step ARIMA Implementation

#### Step 1: Understand the ARIMA Model
- **AR (AutoRegressive):** Models relationship between an observation and its lagged values.
- **I (Integrated):** Differencing the data to make it stationary.
- **MA (Moving Average):** Models relationship between an observation and residual errors.

#### Step 2: Preprocess the Data
- Load historical price data.
- Check for stationarity using the **Augmented Dickey-Fuller (ADF) test**.
- Handle missing values using interpolation.
- Visualize data to identify trends, seasonality, and outliers.

#### Step 3: Identify ARIMA Parameters
- Use **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)**.
- Perform **Grid Search** for optimal (p, d, q) values.

#### Step 4: Train the ARIMA Model
- Split data into training (80%) and testing (20%).
- Fit the model using the **Statsmodels** library in Python.

#### Step 5: Evaluate the Model
- Use RMSE, MAE, and MAPE for evaluation.
- Plot predicted vs actual prices.

#### Step 6: Make Predictions
- Forecast prices for the next 1-2 weeks.
- Provide confidence intervals for uncertainty estimation.
#### Step 7: Interpret and Fine-Tune the Model  
- Analyze residuals to check for patterns and model adequacy.  
- Adjust parameters if necessary to improve performance.  
- Compare ARIMA with other models (e.g., SARIMA, Prophet) for better accuracy.  
- Document findings and potential improvements for future iterations.
---

This document provides a comprehensive overview of our approach to building an AI-ML based model for agri-horticultural price prediction. 🚀
