Importing required libraries and essential packages. 2. Reading the dataset.<br>
#Importing required packages and libraries<br>
   import pandas as pd<br>
   import altair as alt<br>
   import seaborn as sns<br>
   import matplotlib.pyplot as plt<br>
<br>
<br>
#Importing the dataset<br>
   filename="Visual-CW2 dataset.csv" <br>
   data=pd.read_csv(filename)<br>
<br>
<br>
Initial assessment and modification of dataset (with charts)<br>
Printing first two rows of the dataset<br>
#Printing a few datapoints to understand dataset and its columns<br>
   data.head(2)<br>
<br>
<br>
Checking for null values i.e., empty entries<br>
#Checking for null values (empty entries)<br>
   null_values = data.isnull().sum() <br>
   null_values<br>
<br>
<br>
<br>
Statistical description of dataset<br>
#Statistical understanding of the dataset<br>
   data.describe()<br>
Rows, Columns<br>
#Displaying the total number of observations (rows) and the features (columns)<br>
   n_rows,n_cols=data.shape <br>
   print("Rows:",n_rows) <br>
   print("Columns:",n_cols)<br>
<br>
<br>
<br>
Enabling working on large datasets<br>
#MaxRowsError: The number of rows in the dataset exceeds the maximum allowed (5000). This step enables working on large datasets.<br>
   alt.data_transformers.disable_max_rows()<br>
<br>
<br>
<br>
Extracting name of the locations<br>
   data['location'].unique()<br>
<br>
<br>
Adding two more columns to the dataset: year and month<br>
Converting 'sample date' to pandas datetime format<br>
   data['sample date']=pd.to_datetime(data['sample date'])<br>
# Adding additional columns: Creating columns of year and month<br>
   data['year']=data['sample date'].dt.year <br>
   data['month']=data['sample date'].dt.month<br>
<br>
<br>
Bar chart: Number of observations in each location<br>
The first visualization is a simple bar chart created to see how many observations were recorded in each location:<br>
#Total number of readings in each location<br>
#To explicitly consider location as a categorical variable to enable efficient assignment of different colours to each location. <br>
   data['location'] = pd.Categorical(data['location'])<br>
#creating the base bar chart (readings_locations)<br> readings_locations=alt.Chart(data).mark_bar().encode(x="location:N",y="count(value):Q",color="location:N",tooltip['location:N','count(value):Q']).properties(width=400,height=200,title="Total number of readings in each location")<br>
#Creating inference text to add to the base chart<br>
   inference_text = """<br>
         This bar chart shows the total number of sample readings/observations taken from each location present in the dataset.<br>
         Boonsir has the highest number of samples, and Tansanee has the least.<br>
         """<br>
#Creating a dataframe 'text' to store the 'inference_text'<br>
   text=pd.DataFrame({'text': [inference_text]})<br>
#Creating the inference chart that includes the inference text to layer onto the base chart<br>
   inference_chart = alt.Chart(text).mark_text( align='left',baseline='middle', fontSize=12, dx=10, lineBreak='\n').encode( text='text:N')<br>
#Layering the original chart and the inference text chart<br>
   chart_with_inference = readings_locations | inference_chart chart_with_inference<br>
<br>
1. Grouping the dataset by the 'measures' and calculating the associated mean 'value'.<br>
2. 2. Extracting top 5 measures.<br>
# Calculate average values for each measure<br>
   average_values = data.groupby('measure')['value'].mean().reset_index()<br>
# Sorting 'average values' in descending order<br>
   average_values_desc= average_values.sort_values(by='value', ascending=False) print(average_values_desc)<br>
# Isolating the top five highest mean 'value' chemicals/measures<br>
   top_5_measures = average_values_desc.head(5) top_5_measures<br>
#Storing the names of the measures with high mean 'value' in a variable<br>
   top_5_measures_names = top_5_measures['measure'].values<br>
   top_5_measures_names<br>
<br>
<br>
<br>
Visualization of the mean 'value' of the top 5 measures: ['Total dissolved salts', 'Bicarbonates', 'Total hardness','Aluminium', 'Oxygen saturation']:<br>
Pie chart: Mean 'value' for each of the top 5 measures<br>
# Create a pie chart using mark_arc with specified angles<br>
   pie_chart = alt.Chart(top_5_measures).mark_arc().encode( theta='value:O',color='measure:N',tooltip=['measure:N', 'value:Q'] ).properties(width=400,height=400,title="Top Five 
               Highest Chemicals (Pie Chart)")<br>
# Show the chart<br>
   pie_chart<br>
<br>
Filtering the whole data set by storing the detials of the top 5 measures in another variable:<br>
   filtered_data_top5 = data[data['measure'].isin(top_5_measures_names)] <br>
   filtered_data_top5<br>
<br>
(i) MISSING DATA AND (ii) CHANGE IN COLLECTION FREQUENCY <br>
a. Heatmap for the number of observations in each location for the whole dataset<br>
# Pivoting the data for creating a heatmap<br>
   pivoted_heatmap_data = data.pivot_table(values='value', index='location', columns='year', aggfunc='count')<br>
# Create a heatmap with the whole dataset<br>
   plt.figure(figsize=(15, 8))<br>
   sns.heatmap(pivoted_heatmap_data, annot=True, cmap='magma', fmt=".0f", linewidths=.9) <br>
   plt.title('Heatmap of ALL Measures by Location and Year')<br>
   plt.show()<br>
<br>
<br>
b. Creating a heatmap for the filtered data with information for top 5 measures only to show gap in collection years<br>
# Pivoting the data for creating a heatmap<br>
   heatmap_data2 = filtered_data_top5.pivot_table(values='value', index='location', columns='year', aggfunc='count')<br>
# Creating a heatmap with the top five measures<br>
   plt.figure(figsize=(15, 8))<br>
   sns.heatmap(heatmap_data2, annot=True, cmap='magma', fmt=".0f", linewidths=.9) <br>
   plt.title('Heatmap of Top 5 Measures by Location and Year')<br>
   plt.show()<br>
<br>
<br>
c. Data for Aluminium is missing for some of the locations<br>
# No data for measure = Aluminium in some locations<br>
   heatmap_data_aluminium = filtered_data_top5[filtered_data_top5['measure'] == 'Aluminium'].pivot_table(values='value', index='location', columns='year', aggfunc='count') <br>
# Creating a heatmap with the top five measures<br>
   plt.figure(figsize=(3, 3))<br>
   sns.heatmap(heatmap_data_aluminium, annot=True, cmap='viridis', fmt=".0f", linewidths=.9)<br>
   plt.title('Heatmap of Aluminium by Location') <br>
   plt.show()<br>
<br>
<br>
<br>
DUPLICATE/REPEATED VALUES<br>
#Creating a scatter plot to visualize the repeated entries on the same date, taking just one case (at random): the year 2007, and only one measure: Total dissolved salts<br>
   modified_data_rep=data[(data['year']==2007) & (data['measure']=='Total dissolved salts')] <br>
   rep_obs=alt.Chart(modified_data_rep).mark_point(size=200).encode(x='sample date:T',y='value:Q',color='location',tooltip= ['location','value:Q','sample date']).properties(width=900,height=128,title='Total dissolved salts:repeated observations on same date ').facet(row='location') <br>
   rep_obs<br>
<br>
<br>
<br>
ANOMALIES (Iron)<br>
#Line trend (temporal) for the whole dataset<br>
# Convert 'sample date' to DateTime format<br>
# Line chart for mean values over the years for each measure<br>
   line_chart = alt.Chart(data).mark_line(strokeWidth=7).encode( x='year(sample date):O',y='mean(value):Q',color=alt.Color('measure:N', legend=alt.Legend(title='Measure')), tooltip=['measure:N', 'year(sample date):O', 'mean(value):Q']).properties( width=600,height=300,title='Mean Value Trend Over Years' )<br>
   line_chart<br>
<br>
# Line chart for year-wise trend of 'Iron' by region<br>
   iron_trend = alt.Chart(data[data['measure'] == 'Iron']).mark_line(strokeWidth=6).encode( x=alt.X('year:O', title='Year'),y=alt.Y('mean(value):Q', title='Mean Value'),color=alt.Color('location:N', legend=alt.Legend(title='Region')), tooltip=['location:N', 'year:O', 'mean(value):Q'],).properties( width=800,height=400,title='Anomaly: Iron' )<br>
   iron_trend<br>
<br>
<br>
<br>
TRENDS<br>
#Creating the same chart with location as a filter to identify the temporal trends in each location # Create a selection dropdown for locations<br>
   location_dropdown = alt.selection_point(fields=['location'], bind=alt.binding_select(options=filtered_data_top5['location'].unique().tolist()), name='Select Location')<br>
# Create the line chart with the location filter<br>
   top5_trend = alt.Chart(filtered_data_top5).mark_line(strokeWidth=5).encode( x='year:O',y='mean(value):Q',color='measure',tooltip=['measure', 'year:O', 'mean(value):Q']).properties(width=200, height=200).facet(column='measure').add_params( location_dropdown).transform_filter( location_dropdown)<br>
# Display the chart<br>
   top5_trend<br>
<br>
<br>
<br>
OUTLIERS<br>
#Checking normality od the dataset to identify which outlier method to use<br>
   plt.figure(figsize=(8, 3))<br>
   plt.subplot(1, 2, 1)<br>
   sns.histplot(data['value'], bins=20, kde=True, color='red') plt.title('Histogram')<br>
<br>
<br>
# Outlier Detection package<br>
# Storing the original dataset into another variable for further calculations. from scipy.stats import iqr<br>
   data_OD = average_values<br>
<br>
# Calculate the IQR values<br>
   data_OD['iqr'] = iqr(data_OD['value']) # Calculate the range of IQR<br>
   iqr_range = (data_OD['iqr'].min(), data_OD['iqr'].max())<br>
# Display the ranges<br>
   print(data_OD.head(5)) print("IQR Range:", iqr_range)<br>
<br>
<br>
   Q1 = data_OD['value'].quantile(0.25)<br>
   Q3 = data_OD['value'].quantile(0.75)<br>
   IQR = Q3 - Q1<br>
   outliers_iqr = data_OD[(data_OD['value'] < Q1 - 1.5 * IQR) | (data_OD['value'] > Q3 + 1.5 * IQR)] outliers_iqr.head(5)<br>
<br>
   plt.figure(figsize=(7, 7))<br>
   sns.scatterplot(x='value', y='measure', hue='measure', data=outliers_iqr, palette='viridis', s=80) plt.title('Scatter Plot with IQR Outliers')<br>
   plt.xlabel('Years')<br>
   plt.ylabel('Value')<br>
   plt.legend(loc='upper right')<br>
   plt.show()<br>
<br>
# Scatter plot for average values<br>
   plt.figure(figsize=(10, 6))<br>
   sns.scatterplot(x='measure', y='value', data=average_values, label='Average Values')<br>
# Highlight outliers from outliers_iqr DataFrame<br>
   sns.scatterplot(x='measure', y='value', data=outliers_iqr, color='red', marker='x', s=100, label='Outliers')<br>
   plt.title('Scatter Plot of Average Values with Outliers Highlighted')<br>
   plt.xlabel('Measure')<br>
   plt.ylabel('Average Value')<br>
   plt.xticks(rotation=45) # Rotate x-axis labels for better readability plt.tick_params(axis='x', labelsize=4) # Reduce font of x-axis labels for better readability <br>
   plt.legend()<br>
   plt.show()<br>
Above graph is a skewed graph, hence IQR, method is chosen for outlier detection.<br>
<br>
<br>
   plt.figure(figsize=(7, 7))<br>
   sns.scatterplot(x='value', y='measure', hue='measure', data=outliers_iqr, palette='viridis', s=80) plt.title('Scatter Plot with IQR Outliers')<br>
   plt.xlabel('Years')<br>
   plt.ylabel('Value')<br>
   plt.legend(loc='upper right')<br>
   plt.show()<br>
# Scatter plot for average values<br>
   plt.figure(figsize=(10, 6))<br>
   sns.scatterplot(x='measure', y='value', data=average_values, label='Average Values')<br>
# Highlight outliers from outliers_iqr DataFrame<br>
   sns.scatterplot(x='measure', y='value', data=outliers_iqr, color='red', marker='x', s=100, label='Outliers')<br>
   plt.title('Scatter Plot of Average Values with Outliers Highlighted')<br>
   plt.xlabel('Measure')<br>
   plt.ylabel('Average Value')<br>
   plt.xticks(rotation=45) # Rotate x-axis labels for better readability plt.tick_params(axis='x', labelsize=4) # Reduce font of x-axis labels for better readability <br>
   plt.legend()<br>
   plt.show()<br>
   <br>
   outliers_iqr['measure']<br>
Aluminium <br>
Barium<br>
Bicarbonates<br>
Boron<br>
Calcium<br>
Carbonates<br>
Chemical Oxygen Demand (Cr)<br>
Chlorides<br>
Iron<br>
Magnesium<br>
Oxygen saturation<br>
Sodium<br>
Sulphates<br>
Total coliforms<br>
Total dissolved salts<br>
Total hardness<br>
Water temperature<br>



