# dsc232r-amazon-predictive-modeling

Download the dataset and save unzipped files in a folder named 'amazon_data':
https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

In processing our dataset, proper data preparation is crucial to ensure the quality and effectiveness of our analysis. Here are the specific steps we have taken:

### Missing Value Handling
For the essential fields `review_body` and `review_date`, which are critical for our analysis due to their relevance to the review content and timing, we will remove any rows with missing values in these columns. The absence of this information renders the row useless for trend analysis. For missing values in the `product_category` field, since adjacent entries usually belong to the same category, we will employ a forward-fill method to maintain data continuity. Missing values in other fields will be filled based on their data type using the median or mode.

### Data Filtering and Simplification
Given the low proportion of Vine program reviews (only 2,982 out of 523,269), which is too minor to significantly impact our analysis, we have decided to drop this column. Similarly, since all data comes from the US market (the `marketplace` column is always 'US'), this column is redundant and will be removed to save storage space and computational resources.

### Data Time Range Adjustment
Observing that the volume of data increases over the years, likely reflecting the growing base of Amazon users, we will discard data from before 2010. This approach focuses the model training on more representative and relevant data, enhancing the accuracy of predicting future trends.

### Data Encoding and Text Processing
Text category labels, such as `product_category`, will be converted into numerical codes to reduce the complexity of model processing and improve computational efficiency. We will also tokenize the review texts, which are essential for extracting useful information and patterns for text analysis. And we also changed title into vectors and changed text into vectors.

### Dataset Splitting
Data will be split into training and testing sets according to the time order, with the most recent data used as the test set. This setup simulates real-world predictions of product trends, ensuring that the model performs well on unseen data.

Through these meticulous data preprocessing steps, our dataset will be cleaner, more effective, and ready to build an accurate predictive model.

### Extract month and year
Extract these variables as new columns to aid in the modeling process

### Count Product Reviews Per Day
Group by unique product identifier and day to get reviews per day for each product

### Data Modeling 
The review data spans from September 11, 1995, to August 31, 2015. For the initial data modeling phase, we used a year's worth of data to understand the trend of verified product reviews, where "verified" indicates that the product was genuinely purchased by the customer. This initial modeling was designed to forecast the trends for the upcoming months, determining whether the number of reviews was increasing, decreasing, or stable. This analysis helps identify which products were generating more reviews over time and if there were discernible patterns among different products.

The training set consisted of data from 2014, while the test set comprised data from 2015. Upon plotting all product review counts as a reference to verified purchases, we observed that product category number 32 registered the highest number of purchases throughout the year. Although the forthcoming model will be designed to handle multiple categories over multiple years, the initial focus of the data modeling phase is on a single category over a single year to assess the model's accuracy and adaptability.

To forecast the 2015 purchases of products in category 32, we utilized the ARIMA (AutoRegressive Integrated Moving Average) model. This model analyzes the 12 months of data from 2014 to predict the number of reviews in 2015. The ARIMA model aims to predict future values—in this case, purchases by identifying patterns and relationships from historical data.

The subsequent model will build upon the current one, potentially incorporating the use of word2vec data to identify popular buzzwords, which could indicate trends in product popularity.

### Evaluation
In this section, we calculated the Test RMSE and compared the forecast with the testing data and training forecast in the graph. I have written the code for the fitting graph in the milestone3.ipynb file.

However, due to the large size of the dataset, the process is very slow, and the queue time on the server is too long. As a result, we were unable to complete the final part of our code on SDSC. Therefore, I have uploaded a file named milestone3_local on GitHub, which processes a small portion of our dataset and uses this subset to generate the fitting graph. This is mainly to demonstrate that our code works correctly.

### Answer the questions
* Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

Based on the ARIMA model for product_category_num 32 of the year 2014, we forecasted the number of reviews. Using the ARIMA model we saw that the preidction was off from the 
actual data with an RMSE of 28498.15662477966. In the next phase the enchancement of the ARIMA model to add ACF and PACF. ACF autocorrelation function (ACF) helps identify the correlation between observations at different lags (dates) indicating the presence of patterns such as seasonality. While the partial autocorrelation function (PACF) helps determine the extent of correlation between observations while controlling for the influence of previous lags (dates), aiding in identifying the appropriate order of the autoregressive terms. Using the ACF and PACF parameters will be verified against the RMSE to get less error on the forecast. Further improvments of the ARIMA model would be to use the word2vector data to see the primary key words that have the highest count of reviews which we can see which products have similar trends in words that correlate to the review number.

Next models we are thinking of is Linear Regression and SVM. Linear regression we can see which popular products based on star rating will be increasing the number of reviews. This can also help us to see which model would be better to forecast ARIMA or Linear regression. Support Vector Machines (SVM) can assist in understanding which products are most influential in generating reviews through its ability to classify and predict based on labeled data. SVM can classify products based on their attributes (e.g., star ratings, review counts) into categories such as "highly influential," "moderately influential," and "less influential." This classification can be based on historical data that includes product attributes and review counts.

### Conclusion section
* What is the conclusion of your 1st model? What can be done to possibly improve it?
  
The 1st model concluded with an RMSE of 28498.156. This isn't the best given that there is almost a difference of 30k of the forcsted and the actualy 2015 test data for the product_category_num 32. We can increase the training data from 2014 to the range of 2010 to 2014. We do this for all the product_category_num to be able to accurately predict which item is more popular.  By increasing the training data we will be able to catch more underlying patterns thorughout the years. 
