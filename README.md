# Introduction
As our world advances, more activities can be done online through websites, leading many large companies to evolve and dominate various industries. For example, fewer people are going out to watch movies, opting to use services like Netflix instead. E-commerce has also seen a significant rise in popularity in recent years. Large e-commerce companies have been steadily dominating the industry, with platforms like eBay having a long-standing presence and others, such as Etsy, gaining increasing popularity. Additionally, major companies like Best Buy and Walmart are expanding their online presence. Most notably, Amazon is a household name in e-commerce.

By leveraging big data, businesses aim to better grasp customer insights, potentially increasing sales and demand. To achieve this, businesses can use predictive modeling to forecast upcoming trends. These models use historical data to anticipate shifts in consumer preferences, identify popular products, and give businesses a competitive advantage.

In this study, we use historical Amazon data to build two models for predicting trends. Our aim is to develop predictive models that forecast product popularity based on historical sales and word embeddings. This research not only advances predictive analytics but also sheds light on consumer preferences and market dynamics, helping businesses stay ahead in the competitive e-commerce landscape.
# Figures
# Methods
## Data Exploration
### Get Total Number of Observations
```df.count()```
### Get Columns
```df.columns```
### Get Number of Columns
```len(columns)```
### Get Number of Missing Values for Each Column
For each column i```df.filter(df[columns[i]].isNull()).count()```
### Look at Column Types
```df.dtypes```
### Show Column Distributions
```df.describe().show()```
### Check for Duplicates
```df.groupBy(columns).count().where('count > 1')```
### Simple Plot Methods on Columns
#### Histogram
#### Piechart
#### Barplot
#### Boxplot
#### Boxplot with no Outliers
### Simple Visualizations
#### Column `helpful_votes`
#### Column `star_rating`
#### Column `product_category`
#### Yearly Purchase Counts
* Method that allows users to choose a certain year to look at monthly purchasing counts
## Preprocessing
## Model 1
## Model 2
# Results
## Data Exploration
### Get Total Number of Observations
There is a total of 149086 observations
### Get Columns
Columns:
['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date']
### Get Number of Columns
There is a total of 15 columns
### Get Number of Missing Values for Each Column
### Look at Column Types
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
### Show Column Distributions
### Check for Duplicates
### Simple Visualizations
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
## Preprocessing
## Model 1
## Model 2
# Discussion
This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!
# Conclusion
This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts.
# Statement of Collaboration
* Claire Lin
* Jingyi (Alina) Zhou
* Rongrong (Cassy) Xu
* Sol Jung
* Timothy Indrieri
# Final Model and Final Results Summary
Include this in the last paragraph in D.



