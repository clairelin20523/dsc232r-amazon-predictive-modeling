# 1. Introduction
As our world advances, more activities can be done online through websites, leading many large companies to evolve and dominate various industries. For example, fewer people are going out to watch movies, opting to use services like Netflix instead. E-commerce has also seen a significant rise in popularity in recent years. Large e-commerce companies have been steadily dominating the industry, with platforms like eBay having a long-standing presence and others, such as Etsy, gaining increasing popularity. Additionally, major companies like Best Buy and Walmart are expanding their online presence. Most notably, Amazon is a household name in e-commerce.

By leveraging big data, businesses aim to better grasp customer insights, potentially increasing sales and demand. To achieve this, businesses can use predictive modeling to forecast upcoming trends. These models use historical data to anticipate shifts in consumer preferences, identify popular products, and give businesses a competitive advantage.

In this study, we use historical Amazon data to build two models for predicting trends. We aim to develop predictive models that forecast product popularity based on historical sales and word embeddings. This research not only advances predictive analytics but also sheds light on consumer preferences and market dynamics, helping businesses stay ahead in the competitive e-commerce landscape.

# 2. Figures with Media Subcategory
## 2.1 All subcategory trends

<img width="553" alt="1" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/d9b47367-7e60-4593-a327-584d0673940b">

Our main model of interest would be to forecast sale counts for a subcategory of interest, here we look at yearly subcategory sales. We see an increasing trend for all subcategories except for digital media which dropped from 2013 to 2014. The highest subcategory count is for electronics. Outdoor living and home essentials have less sales count but still show an increasing trend.

## 2.2 Media subcategory trend

<img width="567" alt="2" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/e72e990a-bae8-4590-8f38-12f116c76b2b">

We will use the media subcategory as a lasting example in our report, here we have a plot showing the yearly counts for the media subcategory.

## 2.3 Media subcategory category trends

<img width="571" alt="3" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/098ad57d-7335-4aec-82ae-23256c2a9b1b">

Within the media subcategory, we plot the individual categories and look at the trends. Music seems to have the highest sales count while video lowest and barely increasing.

## 2.4 Seasonal Decomposition

<img width="554" alt="4" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/862dd10f-e61b-4f31-a314-3d4fac11ba65">

We see that in the seasonality plot, there seems to be a repeating trend around every two years. However, the whole plot is only for 4 years. Therefore there is no clear conclusion that can be made with these seasonality decompositions.

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
* Yearly Purchase Overview
Method that allows users to choose a certain year to look at monthly purchasing counts
## 3.2 Preprocessing
### 3.2.1 Take care of missing categories
For missing values in the `product_category` field, since adjacent entries usually belong to the same category, we will employ a forward-fill method to maintain data continuity.
### 3.2.2 Remove Columns
Given the low proportion of Vine program reviews (only 2,982 out of 523,269), which is too minor to significantly impact our analysis, we have decided to drop this column. Similarly, since all data comes from the US market (the `marketplace` column is always 'US'), this column is redundant and will be removed to save storage space and computational resources.
### 3.2.3 Filter out rows with missing bodies and dates and verify the purchase
For the essential fields `review_body` and `review_date`, which are critical for our analysis due to their relevance to the review content and timing, we will remove any rows with missing values in these columns. The absence of this information renders the row useless for trend analysis.
### 3.2.4 Filter out old data
Observing that the volume of data increases over the years, likely reflecting the growing base of Amazon users, we will discard data from before 2005. This approach focuses the model training on more representative and relevant data, enhancing the accuracy of predicting future trends.
### 3.2.5 Check other missing values
### 3.2.6 Extract month and year
Extract these variables as new columns to aid in the modeling process
### 3.2.7 Add Sub-category Column
### 3.2.8 Encode Categorical Columns
#### 3.2.8.1 Change title into vectors
#### 3.2.8.1 Change text into vectors
### 3.2.9 Dataset Splitting
Data will be split into training and testing sets according to the time order, with the most recent data used as the test set. This setup simulates real-world predictions of product trends, ensuring that the model performs well on unseen data.
### 3.2.10 Count Product Reviews Per Day
Group by unique product identifier and day to get reviews per day for each product
## 3.3 Model 1 : ARIMA Model
The review data spans from September 11, 1995, to August 31, 2015. For the initial data modeling phase, we used a year's worth of data to understand the trend of verified product reviews, where "verified" indicates that the product was genuinely purchased by the customer. This initial modeling was designed to forecast the trends for the upcoming months, determining whether the number of reviews was increasing, decreasing, or stable. This analysis helps identify which products were generating more reviews over time and if there were discernible patterns among different product categories.

The training set consists of data from 2014, while the test comprises data from 2015. Upon plotting all product review counts as a reference to verified purchases, we observed that product subcategory media registered the highest number of purchases throughout the year. Although the forthcoming model will be designed to handle multiple categories over multiple years, the initial focus of the data modeling phase is on a single category over a single year to assess the model's accuracy and adaptability.

To forecast the 2015 purchases of products in subcategory media, we utilized the ARIMA (AutoRegressive Integrated Moving Average) model. This model analyzes the 12 months of data from 2014 to predict the number of reviews in 2015. The ARIMA model aims to predict future values—in this case, purchases by identifying patterns and relationships from historical data.

## 3.4 Model 2: Logistic Regression
In addition to the ARIMA model, which forecasts trends for upcoming months based on verified purchase reviews, Model 2 is based on the sentiment of the review to predict whether a product will be high in rating indicating popularity. For this model, we analyzed the sentiment of the review_headine text, by creating a list of positive and negative keywords (all in lowercase) as there was too much data in the review text itself. With the new df, we decided that a star rating greater than or equal to three stars is a high rating. 

The following variables are used: product_ category (categorical), helpful_votes, total_votes, and headline_sentiment (numerical). The logistical regression model has two parameters, the featuresCol, and the labelCol “features”. For the features variable “features” which contains the combined feature vector for each row in df ( one-hot encoded vector for categorical variables) and values of each numerical variable. The features column makes predictions about whether an item has a high rating indicating future popularity.

The results are as follows, with 80/20 training and testing, and using the BinaryClassificationEvaluator (labelCol = ‘high_rating’), we got an average of approximately 0.73. This is how accurate the linear regression prediction model is when it comes to predicting whether a product will be popular based on reviews

Insert chart
 Accuracy: 0.7233722543820628
+-----------+----------+--------------------+
|high_rating|prediction|         probability|
+-----------+----------+--------------------+
|          1|       1.0|[0.08732637911517...|
|          1|       1.0|[0.08732637911517...|
…


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
* There are clearly noticeable high outliers

<img width="513" alt="5" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/7461c787-8942-45e2-aa3c-c40edb357f10">

##### Boxplot
* Indeed we see many high outliers
* The boxplot is heavily skewed

<img width="524" alt="6" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/6d80f9dc-3ff8-48d3-a01f-8a0fb8efedca">

#### Column `star_rating`
* Majority of ratings given are 5 stars
* Second most given rating is 1 but still much less than 5
* Increased ratings from 2 to 5 stars

<img width="539" alt="7" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/622b50dc-ff57-42ea-8630-49e86eb1720f">

#### Column `product_category`
*  Look at category distribution through the bar plot

<img width="541" alt="8" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/95d99b95-42be-4deb-9463-1f11ff6d092b">

#### Yearly Purchase Overview

<img width="545" alt="9" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/c908f955-4ab4-4af0-9e0f-2fa021ea211f">

<img width="562" alt="10" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/200b76b5-d988-48a1-9fea-cc24ac0f06c4">

<img width="553" alt="11" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/ed16b70a-9608-472a-bcd1-b673c6652c79">

## 4.2 Preprocessing
### 4.2.1 Take care of missing categories
### 4.2.2 Remove Columns
### 4.2.3 Filter out rows with missing bodies and dates and verify the purchase
### 4.2.4 Filter out old data
### 4.2.5 Check other missing values
### 4.2.6 Extract month and year
### 4.2.7 Add Sub-category Column
### 4.2.8 Encode Categorical Columns
#### 4.2.8.1 Change title into vectors
#### 4.2.8.1 Change text into vectors
### 4.2.9 Dataset Splitting
### 4.2.10 Count Product Reviews Per Day
## 4.3 Model 1With Media Subcategory
### 4.3.1 ACF

<img width="561" alt="12" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/747cfadb-3794-4f8a-8c77-cd2cd55e8d0f">

The Autocorrelation Function (ACF) is a parameter for the ARIMA model. This parameter shows how correlated the values in a time series are with the original series. The ACF then plots the correlation coefficient against the lag. If we are looking at the graph above statistically speaking we should be using a lag of 4 since this seems to have higher significance in terms of the correlation coefficient which means that going back 4 lags will help our model be able to predict better. 

### 4.3.2 PACF

<img width="571" alt="13" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/a99c35a5-f88c-4b1d-9cb5-6b42bad5714a">

The Partial Autocorrelation Function (PACF) is also a parameter of the ARIMA model. The PACF is used to understand the direct value between the past and the current values of a time series.
Unlike ACF which looks at all past values to come up with the correlation, PACF filters out middle values to compare the current value with the value it is currently evaluating to understand the correlation. In the graph above we see that a 1 lag would be the best value to use for the model.

### 4.3.4 Forecast

<img width="585" alt="14" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/f6861ff5-afae-4875-94e2-bffb3447358d">

Based on the ARIMA model for subcategory media of the year 2014, we forecasted the number of reviews. Using the ARIMA model we saw that the prediction was off from the actual data with an RMSE of 28498.15662477966. In the next phase the enhancement of the ARIMA model to add ACF and PACF. ACF autocorrelation function (ACF) helps identify the correlation between observations at different lags (dates) indicating the presence of patterns such as seasonality. The partial autocorrelation function (PACF) helps determine the extent of correlation between observations while controlling for the influence of previous lags (dates), aiding in identifying the appropriate order of the autoregressive terms. Using the ACF and PACF parameters will be verified against the RMSE to get less error on the forecast. Further improvements of the ARIMA model would be to use the word2vector data to see the primary keywords that have the highest count of reviews which we can see which products have similar trends in words that correlate to the review number.

### 4.3.5 Evaluation

<img width="605" alt="15" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/e8900d81-09bc-4725-9257-4589c0334b81">

We see by this chart the forecast in red is the predicted reviews count for the Media data. The blue is our real data. Based on the ARIMA and the parameters from the ACF and PACF we can see our estimation for the values in 2015 for Media review count. This graph shows us that our real data has some fluctuations compared to the test dataset. 

In this section, we calculated the values for Training Forecast, Testing Data, and Forecast across the time series. We also plotted the RMSE values on the training and testing sets as complexity increases. From the graph, it can be observed that the RMSE on the training set increases with rising complexity, but the increase is relatively small compared to that on the testing set. The RMSE on the testing set significantly increases as complexity increases. This suggests that for simple time series models, high complexity is unnecessary.

### 4.3.6 Fitting graph

<img width="568" alt="16" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/dc8e352e-6a01-4e76-bf76-77da80417008">


### 4.3.7 Fitting graph
It differs from typical machine learning models, which generally follow a pattern where the error on the testing set first decreases and then increases with rising complexity. The reason may be that increasing the order of differencing might not help improve the model; instead, it could lead to over-differencing and introduce unnecessary noise. Moreover, increasing the AR component (p) means that the model will consider a longer history of data, while an increase in the MA component (q) means that the model will take into account more recent prediction errors. If the time series data contains relatively more noise, too many AR or MA terms might capture this noise rather than the actual signal. Additionally, in ARIMA models, the choice of parameters is extremely sensitive to the model's performance. Inappropriate parameter selection may directly lead to poor model performance, rather than presenting the overfitting phenomenon of initially decreasing and then increasing error as model complexity increases.

<img width="595" alt="17" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/59724314-ce0d-4d31-9873-b643217ef0d4">


## 4.4 Model 2

<img width="585" alt="18" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/bc15ad67-c0ac-4a6f-8ddf-2ced170b22fb">

We examined the confusion matrix for both the training and testing datasets. This analysis provides a clear view of how well the model predicts binary outcomes by showing the number of correct and incorrect predictions.

<img width="576" alt="19" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/69fb36cd-7a3a-44d1-9c39-a597490362d3">

Secondly, the accuracy and AUC scores summarize the model's performance. Higher values of accuracy and AUC on both training and testing datasets indicate better model performance. If the training accuracy is significantly higher than the testing accuracy, it may suggest overfitting. In our case, the accuracy and AUC on both training and testing datasets are quite close to each other, indicating a well-fitted model.

<img width="542" alt="20" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/2d3bad82-83a8-4c7a-815b-17d3cda004cb">

The ROC curves for both training and testing datasets illustrate the trade-off between true positive rate (TPR) and false positive rate (FPR). These curves provide insights into the model's ability to distinguish between positive and negative classes.

<img width="996" alt="22" src="https://github.com/clairelin20523/dsc232r-amazon-predictive-modeling/assets/113715664/bb35dc72-41db-45dd-bbee-71100b330b3c">

Lastly, for the fitting plot, the first column used was `product_category`, which shows an accuracy score of about 0.67. It is interesting to see that after adding the second column `helpful_votes`, it decreases the accuracy. However, upon adding other columns `total_votes` and `headline_sentiment`, the accuracy increases to about 0.72. When reaching this conclusion, I thought that columnn `helpful_votes` would be noise, however removing it from the whole model decreased the accuracy. Since we have limited columns to include in our model, our model never actually overfits.


# 5. Discussion
## Model 1
The 1st model concluded with an RMSE of 28498.156. This isn't the best given that there is almost a difference of 30k between the forecasted and the actual 2015 test data for the subcategory media. We can increase the training data from 2014 to the range of 2010 to 2014. We do this for all the subcategories to be able to predict which item is more popular accurately. By increasing the training data we will be able to catch more underlying patterns throughout the years.

The next models we are thinking of are Linear Regression and SVM. Linear regression shows which popular products based on star rating will increase the number of reviews. This can also help us to see which model would be better to forecast ARIMA or Linear regression. Support Vector Machines (SVM) can assist in understanding which products are most influential in generating reviews through their ability to classify and predict based on labeled data. SVM can classify products based on their attributes (e.g., star ratings, review counts) into categories such as "highly influential," "moderately influential," and "less influential." This classification can be based on historical data that includes product attributes and review counts.

## Model 2
* What is the conclusion of your 2nd model?
Overall, our logistic regression model demonstrates a good fit in the fitting graph, and considering other models could potentially further enhance our predictive capabilities. The accuracy and AUC scores on both the training and testing datasets are approximately 0.72, indicating a moderately good performance. Additionally, the ROC AUC score of 0.72 suggests that there is a 72% chance that the model will rank a randomly chosen positive instance higher than a randomly chosen negative one. While this score is moderate, it demonstrates that the model has some discriminative power, making it suitable for our task.

* What can be done to possibly improve it?
It would benefit us to include sentiment for the text body as well. However, we fail to do so due to the large memory it takes.

We could investigate why the `helpful_votes` column decreased the accuracy score when used as a predictor with `product_category` but decreased the accuracy when we took it out completely. Maybe there is some multicollinearity between columns that we can investigate.

Our `headline_sentiment` is fairly simple due to it being binary and only using a dictionary of words to conclude, improvement in this column could be done such as using a better dictionary or changing it from binary to decimal. Additionally, using a sophisticated library such as nltk would increase the performance, however, we were unable to do so due to technical and hardware issues.


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

