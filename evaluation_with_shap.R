# ---
# Title: Regression Model Evaluator with SHAP Feature Importance
# Description: This script evaluates multiple regression models for predicting various
#              football metrics and calculates SHAP values to determine feature importance
#              for the Random Forest model.
# ---

# 1. SETUP
# -----------------------------------------------------------------------------
# Define necessary packages. SHAP analysis requires 'fastshap' and 'shapviz'.
packages <- c("caret", "glmnet", "randomForest", "Metrics", "fastshap", "shapviz", "ggplot2")

# Install any missing packages
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load all required libraries
lapply(packages, library, character.only = TRUE)

# Set seed for reproducibility
set.seed(42)


# 2. DATA LOADING AND PREPARATION
# -----------------------------------------------------------------------------
# Read and load data from the source URL
data_url <- "https://raw.githubusercontent.com/MahdiNouraie/Advanced-Football-Analytics-Paper/refs/heads/main/data_aggregated.csv"
data <- read.csv(data_url, header = TRUE, check.names = FALSE)

# Filter data to include only Forwards (FW)
data <- data[data$Pos == 'FW',]

# Define columns to remove, as they are identifiers or redundant
cols_to_rem <- c('Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'MP',
                 'Starts', 'Min', '90s', 'G-PK', 'PKatt', 'CrdY', 'CrdR', 'G+A',
                 'G+A-PK', 'npxG+xA', 'xG+xA', 'Sh', 'SoT', 'G-xG', 'Cmp', 'Att',
                 'TotDist', 'PrgDist', 'Cmp.1', 'A-xA', 'Live', 'Dead', 'Press',
                 'Ground', 'Low', 'High', 'Left', 'Right', 'TI', 'Off', 'SCA',
                 'GCA', 'Tkl+Int', 'Touches', 'Def Pen', 'Carries', 'Targ', 'Rec',
                 'Rec%', 'Prog.1', 'Mn/MP', 'Min%', 'Subs', 'unSub', '+/-',
                 'xG+/-', '2CrdY', 'OG')
data <- data[, setdiff(names(data), cols_to_rem)]

# Scale the entire numeric dataframe
data <- data.frame(scale(data), check.names = FALSE)


# 3. DEFINE PREDICTOR AND RESPONSE VARIABLES
# -----------------------------------------------------------------------------
# Define the 7 representative features chosen as predictors
predictor_variables_raw <- c('Gls', 'onxG', 'Recov', 'Lost', 'Sh/90', 'Sw', 'PPM')

# Clean predictor names to be valid R variable names (e.g., 'Sh/90' becomes 'Sh.90')
predictor_variables_clean <- make.names(predictor_variables_raw)
names(data)[names(data) %in% predictor_variables_raw] <- predictor_variables_clean

# Response variables are all columns that are not predictors
response_variables <- setdiff(colnames(data), predictor_variables_clean)

# Exclude features that were identified as having poor performance
excluded_features <- c('Cmp%', 'SCA90', '+/-90')
response_variables <- setdiff(response_variables, make.names(excluded_features))


# 4. MODEL TRAINING, EVALUATION, AND SHAP ANALYSIS
# -----------------------------------------------------------------------------
# Create a dataframe to store Mean Squared Error (MSE) for each model
df_mse <- data.frame(
  variable = response_variables,
  linear_regression_mse = NA,
  ridge_regression_mse = NA,
  lasso_regression_mse = NA,
  random_forest_mse = NA
)

# Create a list to store SHAP values for each Random Forest model
shap_results_list <- list()

# Loop through each response variable to train models and evaluate performance
for (response_variable in response_variables) {
  # Define predictor (X) and response (y) variables
  X <- data[predictor_variables_clean]
  y <- data[[response_variable]]

  # Split data into training and testing sets (80/20 split)
  train_index <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[train_index, ]
  y_train <- y[train_index]
  X_test <- X[-train_index, ]
  y_test <- y[-train_index]

  # --- Fit the models ---
  model_lr <- lm(y_train ~ ., data = X_train)
  model_ridge <- glmnet(as.matrix(X_train), y_train, alpha = 0)
  model_lasso <- glmnet(as.matrix(X_train), y_train, alpha = 1)
  model_rf <- randomForest(x = X_train, y = y_train, ntree = 100)

  # --- Make predictions ---
  lr_pred <- predict(model_lr, newdata = X_test)
  ridge_pred <- predict(model_ridge, s = 0.01, newx = as.matrix(X_test)) # Using a small lambda
  lasso_pred <- predict(model_lasso, s = 0.01, newx = as.matrix(X_test)) # Using a small lambda
  rf_pred <- predict(model_rf, newdata = X_test)

  # --- Calculate and store MSE ---
  mse_values <- c(
    mse(y_test, lr_pred),
    mse(y_test, ridge_pred),
    mse(y_test, lasso_pred),
    mse(y_test, rf_pred)
  )
  df_mse[df_mse$variable == response_variable, 2:5] <- mse_values

  # --- Calculate and store SHAP values for the Random Forest model ---
  # `fastshap` requires a prediction function that takes model and data
  pfun <- function(model, data) {
    predict(model, newdata = data)
  }
  # Calculate SHAP values on the test set
  shap_values <- fastshap::explain(model_rf, X = X_test, pred_wrapper = pfun, nsim = 50)
  shap_results_list[[response_variable]] <- shap_values
}


# 5. AGGREGATE AND DISPLAY RESULTS
# -----------------------------------------------------------------------------
# --- MSE Summary ---
# Function to calculate summary statistics for MSE
calculate_stats <- function(model_mse_col) {
  col_data <- df_mse[[model_mse_col]]
  data.frame(
    regression_model = gsub("_mse", "", model_mse_col),
    min_mse = min(col_data, na.rm = TRUE),
    q1_mse = quantile(col_data, 0.25, na.rm = TRUE),
    median_mse = median(col_data, na.rm = TRUE),
    mean_mse = mean(col_data, na.rm = TRUE),
    q3_mse = quantile(col_data, 0.75, na.rm = TRUE),
    max_mse = max(col_data, na.rm = TRUE)
  )
}

# Calculate and display statistics for each regression model
model_cols <- c('linear_regression_mse', 'ridge_regression_mse', 'lasso_regression_mse', 'random_forest_mse')
mse_stats_df <- do.call(rbind, lapply(model_cols, calculate_stats))
rownames(mse_stats_df) <- NULL
print("--- MSE Summary Statistics Across All Models ---")
print(mse_stats_df)


# --- SHAP Feature Importance Summary ---
# Combine all SHAP results into a single matrix
agg_shap_values <- do.call(rbind, shap_results_list)

# To create the final plot, we need a corresponding feature matrix
# We'll stack the X_test dataframes from the loop (they are the same each time)
agg_X_test <- do.call(rbind, replicate(length(shap_results_list), X_test, simplify = FALSE))

# Create a shapviz object for visualization
shap_viz_object <- shapviz(agg_shap_values, X = agg_X_test)

# Display the SHAP feature importance plot (overall impact)
print("--- SHAP Feature Importance for Random Forest Models ---")
sv_importance(shap_viz_object, kind = "bar") +
  ggtitle("Aggregated SHAP Feature Importance") +
  theme_minimal()
