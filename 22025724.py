# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as skmet

"""Adfuller is a function imported by this code from the statsmodels.tsa.stattools library.
This function performs the widely used statistical test for stationarity 
of a time series known as the Augmented Dickey-Fuller unit root test.
"""
from statsmodels.tsa.stattools import adfuller

# seasonal_decompose() is used for time series decomposition 

from statsmodels.tsa.seasonal import seasonal_decompose

# Import custom module

import cluster_tools as ct

# Set the option to display all columns

pd.set_option('display.max_columns', None)

columnsName = ["DATE", "value"]

# Read in the "Electric_Production.csv" file as a pandas dataframe and setting the specific columns

df = pd.read_csv("BrentSpotPrice.csv",names = columnsName, header = 0, parse_dates = [0])

#storing in the array format

converted = df.to_numpy()

# Transpose of dataset

transposeddata = converted.T

#print Transpose

print(transposeddata)

df['DATE'] = pd.to_datetime(df['DATE'],infer_datetime_format=True)

# Setting the date as an index

df = df.set_index(['DATE'])

df.head()

# Calculate the rolling mean and rolling std according to months

rolling_mean = df.rolling(window=12).mean()
rolling_std = df.rolling(window=12).std()

# Ploting the rolling the rolling mean and std 

# Setting the figuresize

plt.figure(figsize = (12,8), dpi=300)

# displaying the plot

plt.plot(df, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')

# Setting the x label and y label

plt.xlabel('Date', size = 12)
plt.ylabel('Berent Spot Price', size  = 12)

# Setting the legend at the upper left position

plt.legend(loc = 'upper left')

# setting the super title and title

plt.suptitle('Rolling Statistics', size = 14)
plt.title("22025724")

# Saving the plot

plt.savefig("RollingOutput.png")

# displaying the plot

plt.show()


# Setting the figuresize

plt.figure(figsize=(12,8))

# displaying the plot

plt.plot(df['value'])

# Setting the x label and y label

plt.xlabel("Dates")
plt.ylabel("Berent Spot Price")

# setting the super title and title

plt.suptitle("Berent Spot Price(DPM)")
plt.title("22025724")

# displaying the plot

plt.show()


# Use the augmented Dickey-Fuller test to check for stationarity

adfl = adfuller(df,autolag="AIC")

# Create a DataFrame with ADF test results

output_df = pd.DataFrame({
    "Values": [adfl[0], adfl[1], adfl[2], adfl[3], adfl[4]['1%'], adfl[4]['5%'], adfl[4]['10%']],
    "Metric": ["Test Statistics", "p-value", "No. of lags used", "Number of observations used",               
               "critical value (1%)", "critical value (5%)", "critical value (10%)"]
})


# Print the DataFrame

print(output_df)

# Perform seasonal decomposition of time series data to calculate the useful insights

decompose = seasonal_decompose(df['value'], model='additive', period=7)

# Plot decomposition

decompose.plot()

# Saving and showing the figure

plt.savefig("22025724-Hanan.png",dpi=300)

plt.show()

# Read the dataset from a CSV file

data = pd.read_csv("lifeexpection.csv")

"""Extracting the top five rows using head method."""

data.head()

# filling the null values which can disturb the outcomes

data = data.fillna(0)


# Selecting columns from dataframe

sec_data = data[['1960', '1980', '2000', '2020']]


corr = sec_data.corr()

print(corr)

ct.map_corr(sec_data)

#Saving the plot and showing plot

plt.savefig("heatmap.png")

plt.show()

# Plot a scatter matrix of sec_data

pd.plotting.scatter_matrix(sec_data, figsize=(12, 12), s=5, alpha=0.8)

# Save the scatter matrix plot as an image file

plt.savefig("Matrix.png", dpi=300)

# Display the scatter matrix plot
plt.show()

# Selecting '1960' and '2020' columns from 'sec_data' DataFrame
df_ex = sec_data[['1960', '2020']]

# Dropping rows with null values
df_ex = df_ex.dropna()

# Resetting index
df_ex = df_ex.reset_index()

# Printing first 15 rows of the DataFrame
print(df_ex.iloc[0:15])

# Dropping 'index' column
df_ex = df_ex.drop('index', axis=1)

# Printing first 15 rows of the DataFrame
print(df_ex.iloc[0:15])


# Scale the dataframe
df_norm, df_min, df_max = ct.scaler(df_ex)


print()

print('n  value')

for ncluster in range(2, 10):
    # setting the  cluster with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    #fitting the dataset 
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    
    cen = kmeans.cluster_centers_
    
    print(ncluster, skmet.silhouette_score(df_ex, labels))



# Set number of clusters
ncluster = 4

# Perform KMeans clustering
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Extract x and y coordinates of cluster centers
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]

# Create scatter plot with labeled points and cluster centers
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

# plotting a scatter plot for clusters

plt.scatter(df_norm['1960'], df_norm['2020'], 10, labels, marker='o', cmap=cm)
plt.scatter(xcen, ycen, 45, 'k', marker='d')

# Adding a super title and title

plt.suptitle("Four Clusters", size = 20)
plt.title("21082679",size = 18)

# Adding x labels and y labels

plt.xlabel("Life Expectancy(1970)", size = 16)
plt.ylabel("Life Expectancy(2020)", size = 16)

# Saving the figure
plt.savefig("Four Clusters.png", dpi=300)
plt.show()

# Printing the center

print(cen)

# Applying the backscale function to convert the cluster centre

scen = ct.backscale(cen, df_min, df_max)

print()

print(scen)

xcen = scen[:, 0]
ycen = scen[:, 1]

# cluster by cluster

plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

# plotting a scatter plot for clusters

plt.scatter(df_ex["1960"], df_ex["2020"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")

# Adding a super title and title

plt.suptitle("Four Centered Clusters", size = 20)
plt.title("21082679",size = 18)

# Adding x labels and y labels

plt.xlabel("Life Expectancy(1970)", size = 16)
plt.ylabel("Life Expectancy(2020)", size = 16)

# Saving the figure

plt.savefig("Four Centered Clusters.png", dpi=300)
plt.show()