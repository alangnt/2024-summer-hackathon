import numpy as np
import pandas as pd
import plotly.express as px # plotly.express
import matplotlib.pyplot as plt # matplotlib

# scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler

import seaborn as sns # seaborn

import mpld3 # mpld3
from mpld3 import plugins

import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# All the datasets
df_athletes = pd.read_excel("public/python/datasets/Athletes.xlsx")
df_coaches = pd.read_excel("public/python/datasets/Coaches.xlsx")
df_entries_gender = pd.read_excel("public/python/datasets/EntriesGender.xlsx")
df_medals = pd.read_excel("public/python/datasets/Medals.xlsx")
df_teams = pd.read_excel("public/python/datasets/Teams.xlsx")

df_paris_events = pd.read_csv("public/python/datasets/events.csv")
df_paris_schedules = pd.read_csv("public/python/datasets/schedules.csv")
df_paris_torch_route = pd.read_csv("public/python/datasets/torch_route.csv")
df_paris_venues = pd.read_csv("public/python/datasets/venues.csv")


# Top 10 countries with the most medals in total
top_ten_total_medalists = df_medals.sort_values(by="Total", ascending=False).head(n=10)
top_ten_total_medalists # If we still want to display the data on Jupyter Notebook
most_total_medals_pie = px.pie(top_ten_total_medalists, values="Total", names="Team/NOC")

most_total_medals_pie.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'title': 'Top 10 countries with the most medals in total',
})

most_total_medals_pie.write_html("public/python/graphs/most_total_medals_pie.html")

# Top 10 countries with the most Gold medals
top_ten_total_medalists = df_medals.sort_values(by="Gold", ascending=False).head(n=10)
top_ten_total_medalists # If we still want to display the data on Jupyter Notebook
most_gold_medals_pie = px.pie(top_ten_total_medalists, values="Gold", names="Team/NOC")

most_gold_medals_pie.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'title': 'Top 10 countries with the most Gold medals',
})

most_gold_medals_pie.write_html("public/python/graphs/most_gold_medals_pie.html")

# Top 10 countries with the most Silver medals
top_ten_total_medalists = df_medals.sort_values(by="Silver", ascending=False).head(n=10)
top_ten_total_medalists # If we still want to display the data on Jupyter Notebook
most_silver_medals_pie = px.pie(top_ten_total_medalists, values="Silver", names="Team/NOC")

most_silver_medals_pie.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'title': 'Top 10 countries with the most Silver medals',
})

most_silver_medals_pie.write_html("public/python/graphs/most_silver_medals_pie.html")

# Top 10 countries with the most Bronze medals
top_ten_total_medalists = df_medals.sort_values(by="Bronze", ascending=False).head(n=10)
top_ten_total_medalists # If we still want to display the data on Jupyter Notebook
most_bronze_medals_pie = px.pie(top_ten_total_medalists, values="Bronze", names="Team/NOC")

most_bronze_medals_pie.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'title': 'Top 10 countries with the most Bronze medals',
})

most_bronze_medals_pie.write_html("public/python/graphs/most_bronze_medals_pie.html")

# Calculate counts
athletes_count = df_athletes['NOC'].value_counts().reset_index()
athletes_count.columns = ['NOC', 'Athlete_Count']

coaches_count = df_coaches['NOC'].value_counts().reset_index()
coaches_count.columns = ['NOC', 'Coach_Count']

# Merge counts with medal data
combined_df = pd.merge(athletes_count, coaches_count, on='NOC', how='outer').fillna(0)
medal_counts = df_medals[['Team/NOC', 'Gold', 'Silver', 'Bronze', 'Total']]
medal_counts.columns = ['NOC', 'Gold_Medals', 'Silver_Medals', 'Bronze_Medals', 'Total_Medals']
combined_df = pd.merge(combined_df, medal_counts, on='NOC', how='left').fillna(0)

# Prepare the data
X = combined_df[['Athlete_Count']]
y = combined_df['Total_Medals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Make predictions for 2024 Olympics
combined_df['Predicted_Total_Medals'] = model.predict(combined_df[['Athlete_Count']])
combined_df = combined_df.sort_values('Predicted_Total_Medals', ascending=False)

# Display the top predictions
combined_df[['NOC', 'Athlete_Count', 'Predicted_Total_Medals']].head(10)


# Predicted TOP10 Countries by Predicted Total Medals in 2024 Olympics
top_10_predictions = combined_df.head(10)
linear_regression_bar = px.histogram(top_10_predictions, x="NOC", y="Predicted_Total_Medals")

linear_regression_bar.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'xaxis_showgrid': False,
    'yaxis_showgrid': False,
    'yaxis_title': 'Predicted Total Medals',
    'xaxis_title': 'Country (NOC)',
    'title': 'Top 10 Countries by Predicted Total Medals in 2024 Olympics',
})

linear_regression_bar.write_html('public/python/graphs/linear_regression_bar.html')

# Predicted Number of Athletes vs. Predicted Total Medals
linear_regression_scatter = px.scatter(x=combined_df['Athlete_Count'], y=combined_df['Predicted_Total_Medals'])

linear_regression_scatter.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'xaxis_showgrid': False,
    'yaxis_showgrid': False,
    'yaxis_title': 'Predicted Total Medals',
    'xaxis_title': 'Number of Athletes',
    'title': 'Number of Athletes vs. Predicted Total Medals',
})

linear_regression_scatter.write_html('public/python/graphs/linear_regression_scatter.html')


# Train Using Random Forest Regressor
# Merge counts with medal data
combined_df = pd.merge(athletes_count, coaches_count, on='NOC', how='outer').fillna(0)
medal_counts = df_medals[['Team/NOC', 'Gold', 'Silver', 'Bronze', 'Total']]
medal_counts.columns = ['NOC', 'Gold_Medals', 'Silver_Medals', 'Bronze_Medals', 'Total_Medals']
combined_df = pd.merge(combined_df, medal_counts, on='NOC', how='left').fillna(0)

# Prepare the data with additional features
X = combined_df[['Athlete_Count', 'Coach_Count', 'Gold_Medals', 'Silver_Medals', 'Bronze_Medals']]
y = combined_df['Total_Medals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# Train the model with the best parameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train, y_train)

# Make predictions
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

# Make predictions for 2024 Olympics
combined_df['Predicted_Total_Medals'] = best_rf_model.predict(combined_df[['Athlete_Count', 'Coach_Count', 'Gold_Medals', 'Silver_Medals', 'Bronze_Medals']])
combined_df = combined_df.sort_values('Predicted_Total_Medals', ascending=False)

# Display the top predictions
print(combined_df[['NOC', 'Athlete_Count', 'Coach_Count', 'Predicted_Total_Medals']].head(10))

# Predicted TOP10 Countries by Predicted Total Medals in 2024 Olympics
top_10_predictions = combined_df.head(10)
random_forest_bar = px.histogram(top_10_predictions, x="NOC", y="Predicted_Total_Medals")

random_forest_bar.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'xaxis_showgrid': False,
    'yaxis_showgrid': False,
    'yaxis_title': 'Predicted Total Medals',
    'xaxis_title': 'Country (NOC)',
    'title': 'Top 10 Countries by Predicted Total Medals in 2024 Olympics',
})

random_forest_bar.write_html('public/python/graphs/random_forest_bar.html')

# Predicted Number of Athletes vs. Predicted Total Medals
random_forest_scatter = px.scatter(x=combined_df['Athlete_Count'], y=combined_df['Predicted_Total_Medals'])

random_forest_scatter.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'font_color': 'white',
    'xaxis_showgrid': False,
    'yaxis_showgrid': False,
    'yaxis_title': 'Predicted Total Medals',
    'xaxis_title': 'Number of Athletes',
    'title': 'Number of Athletes vs. Predicted Total Medals',
})

random_forest_scatter.write_html('public/python/graphs/random_forest_scatter.html')