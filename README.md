# 1. Introduction
As our world advances, more activities can be done online through websites, leading many large companies to evolve and dominate various industries. For example, fewer people are going out to watch movies, opting to use services like Netflix instead. E-commerce has also seen a significant rise in popularity in recent years. Large e-commerce companies have been steadily dominating the industry, with platforms like eBay having a long-standing presence and others, such as Etsy, gaining increasing popularity. Additionally, major companies like Best Buy and Walmart are expanding their online presence. Most notably, Amazon is a household name in e-commerce.

By leveraging big data, businesses aim to better grasp customer insights, potentially increasing sales and demand. To achieve this, businesses can use predictive modeling to forecast upcoming trends. These models use historical data to anticipate shifts in consumer preferences, identify popular products, and give businesses a competitive advantage.

In this study, we use historical Amazon data to build two models for predicting trends. Our aim is to develop predictive models that forecast product popularity based on historical sales and word embeddings. This research not only advances predictive analytics but also sheds light on consumer preferences and market dynamics, helping businesses stay ahead in the competitive e-commerce landscape.
# 2. Figures
# 3. Methods
## 3.1 Data Exploration
### 3.1.1 Get Total Number of Observations
```df.count()```
### 3.1.2 Get Columns
```df.columns```
### 3.1.3 Get Number of Columns
```len(columns)```
### 3.1.4 Get Number of Missing Values for Each Column
For each column i```df.filter(df[columns[i]].isNull()).count()```
### 3.1.5 Look at Column Types
```df.dtypes```
### 3.1.6 Show Column Distributions
```df.describe().show()```
### 3.1.7 Check for Duplicates
```df.groupBy(columns).count().where('count > 1')```
### 3.1.8 Simple Plot Methods on Columns
* Histogram
* Piechart
* Barplot
* Boxplot
* Boxplot with no Outliers
### 3.1.9 Simple Visualizations
* Column `helpful_votes`
* Column `star_rating`
* Column `product_category`
* Yearly Purchase Counts
Method that allows users to choose a certain year to look at monthly purchasing counts
## 3.2 Preprocessing
### 3.2.1 Take care of missing categories
For missing values in the `product_category` field, since adjacent entries usually belong to the same category, we will employ a forward-fill method to maintain data continuity.
### 3.2.2 Remove Columns
Given the low proportion of Vine program reviews (only 2,982 out of 523,269), which is too minor to significantly impact our analysis, we have decided to drop this column. Similarly, since all data comes from the US market (the `marketplace` column is always 'US'), this column is redundant and will be removed to save storage space and computational resources.
### 3.2.3 Filter out rows with missing body and date and verified purchase
For the essential fields `review_body` and `review_date`, which are critical for our analysis due to their relevance to the review content and timing, we will remove any rows with missing values in these columns. The absence of this information renders the row useless for trend analysis.
### 3.2.4 Filter out old data
Observing that the volume of data increases over the years, likely reflecting the growing base of Amazon users, we will discard data from before 2005. This approach focuses the model training on more representative and relevant data, enhancing the accuracy of predicting future trends.
### 3.2.5 Check other missing values
### 3.2.6 Extract month and year
### 3.2.7 Add Sub-category Column
### 3.2.8 Encode Categorical Columns
#### 3.2.8.1 Change title into vectors
#### 3.2.8.1 Change text into vectors
### 3.2.9 Dataset Splitting
Data will be split into training and testing sets according to the time order, with the most recent data used as the test set. This setup simulates real-world predictions of product trends, ensuring that the model performs well on unseen data.
### 3.2.10 Count Product Reviews Per Day
## 3.3 Model 1
## 3.4 Model 2
# 4. Results
## 4.1 Data Exploration
### 4.1.1 Get Total Number of Observations
There is a total of 149086 observations
### 4.1.2 Get Columns
Columns:
['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date']
### 4.1.3 Get Number of Columns
There is a total of 15 columns
### 4.1.4 Get Number of Missing Values for Each Column
### 4.1.5 Look at Column Types
'marketplace' column is of type 'string'
'customer_id' column is of type 'int'
'review_id' column is of type 'string'
'product_id' column is of type 'string'
'product_parent' column is of type 'int'
'product_title' column is of type 'string'
'product_category' column is of type 'string'
'star_rating' column is of type 'int'
'helpful_votes' column is of type 'int'
'total_votes' column is of type 'int'
'vine' column is of type 'string'
'verified_purchase' column is of type 'string'
'review_headline' column is of type 'string'
'review_body' column is of type 'string'
'review_date' column is of type 'timestamp'
### 4.1.6 Show Column Distributions
### 4.1.7 Check for Duplicates
### 4.1.8 Simple Plot Methods on Columns
### 4.1.9 Simple Visualizations
#### Column `helpful_votes`
##### Histogram
* There are clearly noticeable high outliers, which conceal the visibility of the distribution
##### Boxplot
* Indeed we see many high outliers
##### Boxplot with No Outliers
* Outliers removed box plot, we see most reviews have no votes
#### Column `star_rating`
* Majority of ratings given are 5 stars
* Second most given rating is 1 but still much less than 5
* Increased ratings from 2 to 5 stars
#### Column `product_category`
* Look at category distribution through pie chart
#### Yearly Purchase Counts
## 4.2 Preprocessing
### 4.2.1 Take care of missing categories
### 4.2.2 Remove Columns
### 4.2.3 Filter out rows with missing body and date and verified purchase
### 4.2.4 Filter out old data
### 4.2.5 Check other missing values
### 4.2.6 Extract month and year
### 4.2.7 Add Sub-category Column
### 4.2.8 Encode Categorical Columns
#### 4.2.8.1 Change title into vectors
#### 4.2.8.1 Change text into vectors
### 4.2.9 Dataset Splitting
### 4.2.10 Count Product Reviews Per Day
## 4.3 Model 1
## 4.4 Model 2
# 5. Discussion
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

# 6. Conclusion
In this study, we aimed to leverage historical Amazon data to build predictive models for forecasting product popularity. By utilizing the ARIMA model for time series forecasting and the logistic regression model for classification, we have demonstrated the potential of predictive analytics in understanding consumer preferences and market trends. However, there are several reflections and potential future directions that could enhance this research.

While the ARIMA and logistic regression models provided valuable insights, exploring more advanced models such as LSTM (Long Short-Term Memory) networks for time series data or ensemble methods like Random Forests and Gradient Boosting for classification could enhance predictive performance. These models can capture more complex patterns and interactions within the data.

One exciting direction for future research is the development of real-time predictive analytics. Implementing streaming data processing frameworks could enable businesses to make real-time predictions and adapt quickly to changing consumer trends. This could be particularly valuable for dynamic pricing strategies and inventory management. Beyond the technical aspects, integrating predictive models with actual business strategies is crucial. Future research could focus on case studies where businesses successfully leverage these models to improve decision-making processes, customer satisfaction, and overall profitability.

Our study highlights the potential of predictive modeling in e-commerce, particularly in understanding and forecasting product popularity. While our models demonstrated moderate success, there is significant room for improvement and expansion. By addressing the limitations and exploring more advanced techniques, future research can further enhance the predictive capabilities and provide deeper insights into consumer behavior.


# 7. Statement of Collaboration
* Claire Lin: Worked on abstract draft 2, data preprocessing, data exploration, organizing notebooks, and the final report, and had Zoom calls with multiple team members.
* Jingyi (Alina) Zhou: Worked on the preprocessing writeup and the evaluation of model 1 with its writeup.
* Rongrong (Cassy) Xu: Worked on the first draft of the abstract, combined notebooks and submitted data exploration, evaluated model 2 with its writeup, and participated in editing the final report.
* Sol Jung: Proofread the abstract and final report, developed the initial models for model 1 and model 2, improved model 2, and helped write the related writeup for the models i.e. ARIMA and Logistic Regression, zoom calls with multiple team members.
* Timothy Indrieri: Helped Program  Model 1 and Model 2, and helped write the related writeup for the models i.e. ARIMA and Logistic Regression, zoom calls with multiple team members.

# 8. Final Model and Final Results Summary
Model 1: ARIMA Model
The ARIMA (AutoRegressive Integrated Moving Average) model was utilized to forecast the number of verified product reviews, spanning from September 11, 1995, to August 31, 2015. For the initial phase, we focused on the year 2014 to predict trends for 2015. Our analysis centered on verified product reviews to determine patterns of increasing, decreasing, or stable review counts across different product categories.

Results:
The ARIMA model analyzed 12 months of 2014 data to predict 2015 review counts.
The model identified subcategory media as having the highest number of purchases throughout the year.
The Root Mean Squared Error (RMSE) for the ARIMA model predictions was 28,498.16, indicating a significant deviation from actual review counts.

Model 2: Logistic Regression
The logistic regression model aimed to predict product popularity based on review sentiments. This model leveraged the sentiment of review headlines, along with other features, to classify products as having high or low ratings. A star rating of three or more indicated a high rating.

Results:
An 80/20 train-test split was used to evaluate the model.
The model achieved an accuracy and AUC score of approximately 0.73 on both training and testing datasets.
The ROC AUC score of 0.72 suggests a moderate ability to discriminate between high and low-rated products, with a 72% chance of correctly ranking a randomly chosen positive instance over a negative one.

