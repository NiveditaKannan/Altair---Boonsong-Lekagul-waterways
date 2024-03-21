Importing required libraries and essential packages. 2. Reading the dataset.
#Importing required packages and libraries
   import pandas as pd
   import altair as alt
   import seaborn as sns
   import matplotlib.pyplot as plt


#Importing the dataset
   filename="Visual-CW2 dataset.csv" 
   data=pd.read_csv(filename)


Initial assessment and modification of dataset (with charts)
Printing first two rows of the dataset
#Printing a few datapoints to understand dataset and its columns
   data.head(2)


Checking for null values i.e., empty entries
#Checking for null values (empty entries)
   null_values = data.isnull().sum() 
   null_values

id             0
value          0
location       0
sample date    0
measure        0
dtype: int64


Statistical description of dataset
#Statistical understanding of the dataset
   data.describe()
              id             value 
count    1.368240e+05    136824.000000
mean     1.197736e+06     24.021591
std      1.000893e+06     231.254038
min      2.221000e+03      0.000000
25%      5.147528e+05      0.059950
50%      8.891045e+05      1.862100
75%      1.640213e+06     14.100000
max      3.448888e+06     37959.280000

Rows, Columns
#Displaying the total number of observations (rows) and the features (columns)
   n_rows,n_cols=data.shape 
   print("Rows:",n_rows) 
   print("Columns:",n_cols)

Rows: 136824
Columns: 5


Enabling working on large datasets
#MaxRowsError: The number of rows in the dataset exceeds the maximum allowed (5000). This step enables working on large datasets.
   alt.data_transformers.disable_max_rows()

DataTransformerRegistry.enable('default')


Extracting name of the locations
   data['location'].unique()

array(['Boonsri', 'Kannika', 'Chai', 'Kohsoom', 'Somchair', 'Sakda', 'Busarakhan', 'Tansanee', 'Achara', 'Decha'], dtype=object)


Adding two more columns to the dataset: year and month
Converting 'sample date' to pandas datetime format
   data['sample date']=pd.to_datetime(data['sample date'])
# Adding additional columns: Creating columns of year and month
   data['year']=data['sample date'].dt.year 
   data['month']=data['sample date'].dt.month


Bar chart: Number of observations in each location
The first visualization is a simple bar chart created to see how many observations were recorded in each location:
#Total number of readings in each location
#To explicitly consider location as a categorical variable to enable efficient assignment of different colours to each location. data['location'] = pd.Categorical(data['location'])
#creating the base bar chart (readings_locations)
   readings_locations= alt.Chart(data).mark_bar().encode( x="location:N",y="count(value):Q",color="location:N", tooltip=['location:N','count(value):Q']).properties(width=400,height=200,title="Total number of readings in each location")
#Creating inference text to add to the base chart
   inference_text = """
         This bar chart shows the total number of sample readings/observations taken from each location present in the dataset.
         Boonsir has the highest number of samples, and Tansanee has the least.
         """
#Creating a dataframe 'text' to store the 'inference_text'
   text=pd.DataFrame({'text': [inference_text]})
#Creating the inference chart that includes the inference text to layer onto the base chart
   inference_chart = alt.Chart(text).mark_text( align='left',baseline='middle', fontSize=12, dx=10, lineBreak='\n').encode( text='text:N')
#Layering the original chart and the inference text chart
   chart_with_inference = readings_locations | inference_chart chart_with_inference

1. Grouping the dataset by the 'measures' and calculating the associated mean 'value'.
2. 2. Extracting top 5 measures.
# Calculate average values for each measure
   average_values = data.groupby('measure')['value'].mean().reset_index()
# Sorting 'average values' in descending order
   average_values_desc= average_values.sort_values(by='value', ascending=False) print(average_values_desc)
# Isolating the top five highest mean 'value' chemicals/measures
   top_5_measures = average_values_desc.head(5) top_5_measures
#Storing the names of the measures with high mean 'value' in a variable
   top_5_measures_names = top_5_measures['measure'].values
   top_5_measures_names
array(['Total dissolved salts', 'Bicarbonates', 'Total hardness','Aluminium', 'Oxygen saturation'], dtype=object)


Visualization of the mean 'value' of the top 5 measures: ['Total dissolved salts', 'Bicarbonates', 'Total hardness','Aluminium', 'Oxygen saturation']:
Pie chart: Mean 'value' for each of the top 5 measures
# Create a pie chart using mark_arc with specified angles
   pie_chart = alt.Chart(top_5_measures).mark_arc().encode( theta='value:O',color='measure:N',tooltip=['measure:N', 'value:Q'] ).properties(width=400,height=400,title="Top Five Highest Chemicals (Pie Chart)")
# Show the chart
   pie_chart

Filtering the whole data set by storing the detials of the top 5 measures in another variable:
   filtered_data_top5 = data[data['measure'].isin(top_5_measures_names)] 
   filtered_data_top5

(i) MISSING DATA AND (ii) CHANGE IN COLLECTION FREQUENCY 
a. Heatmap for the number of observations in each location for the whole dataset
# Pivoting the data for creating a heatmap
   pivoted_heatmap_data = data.pivot_table(values='value', index='location', columns='year', aggfunc='count')
# Create a heatmap with the whole dataset
   plt.figure(figsize=(15, 8))
   sns.heatmap(pivoted_heatmap_data, annot=True, cmap='magma', fmt=".0f", linewidths=.9) 
   plt.title('Heatmap of ALL Measures by Location and Year')
   plt.show()


b. Creating a heatmap for the filtered data with information for top 5 measures only to show gap in collection years
# Pivoting the data for creating a heatmap
   heatmap_data2 = filtered_data_top5.pivot_table(values='value', index='location', columns='year', aggfunc='count')
# Creating a heatmap with the top five measures
   plt.figure(figsize=(15, 8))
   sns.heatmap(heatmap_data2, annot=True, cmap='magma', fmt=".0f", linewidths=.9) 
   plt.title('Heatmap of Top 5 Measures by Location and Year')
   plt.show()


c. Data for Aluminium is missing for some of the locations
# No data for measure = Aluminium in some locations
   heatmap_data_aluminium = filtered_data_top5[filtered_data_top5['measure'] == 'Aluminium'].pivot_table(values='value', index='location', columns='year', aggfunc='count') 
# Creating a heatmap with the top five measures
   plt.figure(figsize=(3, 3))
   sns.heatmap(heatmap_data_aluminium, annot=True, cmap='viridis', fmt=".0f", linewidths=.9)
   plt.title('Heatmap of Aluminium by Location') 
   plt.show()



DUPLICATE/REPEATED VALUES
#Creating a scatter plot to visualize the repeated entries on the same date, taking just one case (at random): the year 2007, and only one measure: Total dissolved salts
   modified_data_rep=data[(data['year']==2007) & (data['measure']=='Total dissolved salts')] 
   rep_obs=alt.Chart(modified_data_rep).mark_point(size=200).encode(x='sample date:T',y='value:Q',color='location',tooltip= ['location','value:Q','sample date']).properties(width=900,height=128,title='Total dissolved salts:repeated observations on same date ').facet(row='location') 
   rep_obs



ANOMALIES (Iron)
#Line trend (temporal) for the whole dataset
# Convert 'sample date' to DateTime format
# Line chart for mean values over the years for each measure
   line_chart = alt.Chart(data).mark_line(strokeWidth=7).encode( x='year(sample date):O',y='mean(value):Q',color=alt.Color('measure:N', legend=alt.Legend(title='Measure')), tooltip=['measure:N', 'year(sample date):O', 'mean(value):Q']).properties( width=600,height=300,title='Mean Value Trend Over Years' )
   line_chart

# Line chart for year-wise trend of 'Iron' by region
   iron_trend = alt.Chart(data[data['measure'] == 'Iron']).mark_line(strokeWidth=6).encode( x=alt.X('year:O', title='Year'),y=alt.Y('mean(value):Q', title='Mean Value'),color=alt.Color('location:N', legend=alt.Legend(title='Region')), tooltip=['location:N', 'year:O', 'mean(value):Q'],).properties( width=800,height=400,title='Anomaly: Iron' )
   iron_trend



TRENDS
#Creating the same chart with location as a filter to identify the temporal trends in each location # Create a selection dropdown for locations
   location_dropdown = alt.selection_point(fields=['location'], bind=alt.binding_select(options=filtered_data_top5['location'].unique().tolist()), name='Select Location')
# Create the line chart with the location filter
   top5_trend = alt.Chart(filtered_data_top5).mark_line(strokeWidth=5).encode( x='year:O',y='mean(value):Q',color='measure',tooltip=['measure', 'year:O', 'mean(value):Q']).properties(width=200, height=200).facet(column='measure').add_params( location_dropdown).transform_filter( location_dropdown)
# Display the chart
   top5_trend



OUTLIERS
#Checking normality od the dataset to identify which outlier method to use
   plt.figure(figsize=(8, 3))
   plt.subplot(1, 2, 1)
   sns.histplot(data['value'], bins=20, kde=True, color='red') plt.title('Histogram')


# Outlier Detection package
# Storing the original dataset into another variable for further calculations. from scipy.stats import iqr
   data_OD = average_values

# Calculate the IQR values
   data_OD['iqr'] = iqr(data_OD['value']) # Calculate the range of IQR
   iqr_range = (data_OD['iqr'].min(), data_OD['iqr'].max())
# Display the ranges
   print(data_OD.head(5)) print("IQR Range:", iqr_range)


   Q1 = data_OD['value'].quantile(0.25)
   Q3 = data_OD['value'].quantile(0.75)
   IQR = Q3 - Q1
   outliers_iqr = data_OD[(data_OD['value'] < Q1 - 1.5 * IQR) | (data_OD['value'] > Q3 + 1.5 * IQR)] outliers_iqr.head(5)

   plt.figure(figsize=(7, 7))
   sns.scatterplot(x='value', y='measure', hue='measure', data=outliers_iqr, palette='viridis', s=80) plt.title('Scatter Plot with IQR Outliers')
   plt.xlabel('Years')
   plt.ylabel('Value')
   plt.legend(loc='upper right')
   plt.show()

# Scatter plot for average values
   plt.figure(figsize=(10, 6))
   sns.scatterplot(x='measure', y='value', data=average_values, label='Average Values')
# Highlight outliers from outliers_iqr DataFrame
   sns.scatterplot(x='measure', y='value', data=outliers_iqr, color='red', marker='x', s=100, label='Outliers')
   plt.title('Scatter Plot of Average Values with Outliers Highlighted')
   plt.xlabel('Measure')
   plt.ylabel('Average Value')
   plt.xticks(rotation=45) # Rotate x-axis labels for better readability plt.tick_params(axis='x', labelsize=4) # Reduce font of x-axis labels for better readability plt.legend()
   plt.show()
Above graph is a skewed graph, hence IQR, method is chosen for outlier detection.
                  measure     value       iqr
0  1,2,3-Trichlorobenzene  0.001320  5.483461
1  1,2,4-Trichlorobenzene  0.001000  5.483461

   plt.figure(figsize=(7, 7))
   sns.scatterplot(x='value', y='measure', hue='measure', data=outliers_iqr, palette='viridis', s=80) plt.title('Scatter Plot with IQR Outliers')
   plt.xlabel('Years')
   plt.ylabel('Value')
   plt.legend(loc='upper right')
   plt.show()
# Scatter plot for average values
   plt.figure(figsize=(10, 6))
   sns.scatterplot(x='measure', y='value', data=average_values, label='Average Values')
# Highlight outliers from outliers_iqr DataFrame
   sns.scatterplot(x='measure', y='value', data=outliers_iqr, color='red', marker='x', s=100, label='Outliers')
   plt.title('Scatter Plot of Average Values with Outliers Highlighted')
   plt.xlabel('Measure')
   plt.ylabel('Average Value')
   plt.xticks(rotation=45) # Rotate x-axis labels for better readability plt.tick_params(axis='x', labelsize=4) # Reduce font of x-axis labels for better readability plt.legend()
   plt.show()
   
   outliers_iqr['measure']
Aluminium 
Barium
Bicarbonates
Boron
Calcium
Carbonates
Chemical Oxygen Demand (Cr)
Chlorides
Iron
Magnesium
Oxygen saturation
Sodium
Sulphates
Total coliforms
Total dissolved salts
Total hardness
Water temperature



