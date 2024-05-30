# dsc232r-amazon-predictive-modeling

Download the dataset and save unzipped files in a folder named 'amazon_data':
https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

In processing our dataset, proper data preparation is crucial to ensure the quality and effectiveness of our analysis. Here are the specific steps we have taken:

### Missing Value Handling
For the essential fields `review_body` and `review_date`, which are critical for our analysis due to their relevance to the review content and timing, we will remove any rows with missing values in these columns. The absence of this information renders the row useless for trend analysis. For missing values in the `product_category` field, since adjacent entries usually belong to the same category, we will employ a forward-fill method to maintain data continuity. Missing values in other fields will be filled based on their data type using the median or mode.

### Data Filtering and Simplification
Given the low proportion of Vine program reviews (only 2,982 out of 523,269), which is too minor to significantly impact our analysis, we have decided to drop this column. Similarly, since all data comes from the US market (the `marketplace` column is always 'US'), this column is redundant and will be removed to save storage space and computational resources.

### Data Time Range Adjustment
Observing that the volume of data increases over the years, likely reflecting the growing base of Amazon users, we will discard data from before 2005. This approach focuses the model training on more representative and relevant data, enhancing the accuracy of predicting future trends.

### Data Encoding and Text Processing
Text category labels, such as `product_category`, will be converted into numerical codes to reduce the complexity of model processing and improve computational efficiency. We will also tokenize the review texts, which are essential for extracting useful information and patterns for text analysis.

### Anomaly Data Handling
We will identify and remove reviews likely generated by automated scripts, as such data typically feature monotonous sentiments and high repetition rates, which could affect the model's accuracy and generalizability.

### Feature Standardization
We will standardize or normalize numerical features, especially when using distance-based models, to optimize model performance and predictive capabilities.

### Dataset Splitting
Data will be split into training and testing sets according to the time order, with the most recent data used as the test set. This setup simulates real-world predictions of product trends, ensuring that the model performs well on unseen data.

Through these meticulous data preprocessing steps, our dataset will be cleaner, more effective, and ready to build an accurate predictive model.

# Introduction
As our world advances, more activities can be done online through websites, leading many large companies to evolve and dominate various industries. For example, fewer people are going out to watch movies, opting to use services like Netflix instead. E-commerce has also seen a significant rise in popularity in recent years. Large e-commerce companies have been steadily dominating the industry, with platforms like eBay having a long-standing presence and others, such as Etsy, gaining increasing popularity. Additionally, major companies like Best Buy and Walmart are expanding their online presence. Most notably, Amazon is a household name in e-commerce.

By leveraging big data, businesses aim to better grasp customer insights, potentially increasing sales and demand. To achieve this, businesses can use predictive modeling to forecast upcoming trends. These models use historical data to anticipate shifts in consumer preferences, identify popular products, and give businesses a competitive advantage.

In this study, we use historical Amazon data to build two models for predicting trends. Our aim is to develop predictive models that forecast product popularity based on historical sales and word embeddings. This research not only advances predictive analytics but also sheds light on consumer preferences and market dynamics, helping businesses stay ahead in the competitive e-commerce landscape.
# Figures
# Methods
## Data Exploration
### Get Total Number of Observations
## Preprocessing
## Model 1
## Model 2
# Results
##Data Exploration
##Preprocessing
##Model 1
##Model 2
#Discussion
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!
#Conclusion
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts.
#Statement of Collaboration
*Claire Lin
*Jingyi (Alina) Zhou
*Rongrong (Cassy) Xu
*Sol Jung
*Timothy Indrieri
Final Model and Final Results Summary
Include this in the last paragraph in D.




