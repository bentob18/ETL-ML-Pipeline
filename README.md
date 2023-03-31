# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

    .
    ├── app     
    │   ├── run.py                            # Flask file that runs app
    │   └── templates   
    │       ├── go.html                       # Classification result page of web app
    │       └── master.html                   # Main page of web app    
    ├── data                   
    │   ├── categories.csv                    # Dataset including all the categories  
    │   ├── messages.csv                      # Dataset including all the messages
    │   ├── ETL Pipeline Preparation.ipynb    # ETL Notebook
    │   ├──process_data.py                    # Data cleaning
    │   └──disaster_responde_db.db            # Database cleaned and saved
    ├── models
    │   ├──  train_classifier.py              # Train ML model        
    │   ├── ML Pipeline Preparation.ipynb     # ML notebook
    │   └── classifier.pkl                    # Pickle model
    └── README.md
# Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following sections:

1. Processing data, building an ETL pipeline to extract data, clean the data and save them in a SQLite DB.
2. Build a machine learning pipeline to train the which can classify text message in various categories.
3. Run a web app that can show model results in real time.
# Installation
For this project you will need to install some libraries like:
1. NumPy
2. Pandas
3. Scikit-Learn
4. NLTK
5. SQLalchemy
6. Picle

# Executing
You can run the following commands in the project's directory to set up the database, train model and save the model.

To run ETL pipeline to clean data and store the processed data in the database:
1. python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv disaster_response_db.db
To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file:
2. python models/train_classifier.py disaster_response_db.db models/classifier.pkl
Run the following command in the app's directory to run your web app:
3. python app/run.py

Go to http://0.0.0.0:3000/

# Screenshots of visualization
![Captura de tela 2023-03-31 175631](https://user-images.githubusercontent.com/103281382/229228783-d053bb78-2813-4d6b-b2d9-194393e162b8.png)
![Captura de tela 2023-03-31 175653](https://user-images.githubusercontent.com/103281382/229228820-7e5f5995-0e08-44cb-946a-eac142837de0.png)
![Captura de tela 2023-03-31 175703](https://user-images.githubusercontent.com/103281382/229228839-39e63a66-5f92-469a-8ca2-09b6bd6afaee.png)
![Captura de tela 2023-03-31 180003](https://user-images.githubusercontent.com/103281382/229229143-9284cd03-cf6a-44b7-82ea-919b50ea1eca.png)

# Acknowledgements
1. Udacity for providing an amazing Data Science Nanodegree Program
2. Figure Eight for providing the relevant dataset to train the model
