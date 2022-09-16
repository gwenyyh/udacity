# Disaster Response Pipeline Project

The main goal of this project is to classify messages into categories in order to help emergency agencies to respond during disaster events. 

The training dataset from [Figure Eight](https://appen.com/) uses real messages sent during the disasters' happenings. 

On the web app you can classify a message (text) as well as explore a little about the training set.

### Instructions to run:
1. Go to the `disaster_response` directory to run the following commands for this project:

    `cd disaster_response`

2. To ensure you have correct environment to run the project, please run the following to install all the packages needed: 

    `pip install -r requirements.txt`

2. Run the following commands to set up your database and model:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model
        `python models/train_classifier.py sqlite:///DisasterResponse.db models/random_forest.pkl`

3. Run the following command to run your web app:
    `python app/run.py`

4. Go to the url returned in the command line

### Files:
- app  
    | - template  
    | |- master.html # main page of web app  
    | |- go.html # classification result page of web app  
    |- run.py # Flask file that runs app  

- data  
    |- disaster_categories.csv # data to process  
    |- disaster_messages.csv # data to process  
    |- process_data.py # script to process the raw data into clean data  
    |- DisasterResponse.db # database to save clean data to  

- models  
    |- train_classifier.py  # script to train, evaluate and save the model  
    |- random_forest.pkl # saved model with random forest and the one used in the web app  
    |- decision_tree.pkl # saved model with decision tree 
    |- kneighbors.pkl # saved model with k nearest neighbors   
    |- logistic_regression.pkl # saved model with logistic regression  
    |- model_with_genre.pkl # saved model with random forest and feature genre  

- explore_notebook.ipynb # notebook with exploration work
- README.md