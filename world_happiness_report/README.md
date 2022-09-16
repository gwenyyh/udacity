# Introduction

This Jupyter Notebook demonstrates some EDA on the world happiness report, especially focused on the contribution of GDP per Capita and Healthy Life Expectancy to Life Ladder Score, trying to answer the following questions:

- how money, health and happiness are related to each other
- which is more important, money or health, when it comes to predicting happiness
- how well can the two predict happiness

More details please read my medium blog <a href="https://yinghuiyang-63486.medium.com/what-brings-happiness-money-or-health-5b6dff06cc6">What Brings Happiness, Money or Health?</a>. 

# Libraries used

`pandas` `numpy` `sklearn` `seaborn` `matplotlib`  

# Motivation for the project

This project is part of my Udacity's nano program in data science - Write a Data Science Blog Post. As a person who is constantly thinking about life questions, I have always wondered what makes people happy. While exploring datasets on Kaggle, I found that the data from world happiness report could potentially reveal some facts in happiness on a macro level.


# Files contained

- /data folder: contains 2 datasets used in csv formats.  
- world_happiness_report notebook: contains the main analysis on the datasets. 
- functions notebook: contains functions used in the world_happiness_report notebook.  

# Summary of the Analysis

- Money, health and happiness are overall all positively correlated to each other.
- However within a specific region, health plays little role to happiness especially for the very poor regions while money appears to be important in every region.
- When predicting happiness money is way more significant than health but overall they can only explain part of people’s happiness.
