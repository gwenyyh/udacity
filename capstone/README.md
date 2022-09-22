# Sparkify Project

The main goal of this project is to perform data manipulation, analysis and make churn predictions with customer data using Pyspark.

The results will be shown in a web app.

### Instructions to run the web app:
1. Go to the `disaster_response` directory to run the following commands for this project:

    `cd disaster_response`

2. To ensure you have correct environment to run the project, please run the following to install all the packages needed: 

    `pip install -r requirements.txt`

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
    |- mini_sparkify_event_data1.json # raw dataset part 1 due to size limit in git repo. It contains event data of customer visits for a music streaming company.  
    |- mini_sparkify_event_data2.json # raw dataset pat 2 

- gbt # saved Pyspark model, which should be read with entire folder 
- Sparkify.ipynb # notebook with major work
- README.md
- requirements.txt