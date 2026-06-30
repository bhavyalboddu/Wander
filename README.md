# Wander

#### This project was submitted to Hack Downtown 2025 @ UVA

Link to Devpost: https://devpost.com/software/wander-vjhdyn

NOTE: This project was developed and evaluated inside Google Colab to utilize cloud GPU resources. The final production-ready scripts and notebooks have been consolidated here for version control and portfolio presentation.

This application, Wander, is designed to revolutionize the way travelers explore new cities by leveraging machine learning. We have developed a machine learning model that curates a list of activities recommended to you to do in a new city based on past tourist attractions you have liked or disliked in other cities. By collecting data on tourist attractions across various cities and integrating user preferences, we provide a tailored experience that evolves with every trip.

We first select the city we are exploring. Now we are presented with a curated list of attractions from our database. We select which ones we would like to save for a future visit, and which ones we are not interested in based on a bunch of different factors. This feedback is used to train our recommender model that identifies patterns in user preferences. The model learns from each selection to refine recommendations for future trips. Finally, we can see all of our “liked” locations here!

## Technologies Used
* Anvil
* Python
* Scikit-learn
* Tensorflow

## How we built this application

Since the application depends heavily on user-generated data, we used Generative AI to curate a list of tourist attractions in various cities including features such as rating, cost, address, suitability, and estimated time. Once of our team members, acted as a user of our app, deciding whether or not they would visit these stops. Based off of this, we were able to train multiple machine learning models (Logistic Regression, Decision Trees, Random Forest, Neural Networks, and CNN) and found that CNN resulted in 82 percent. However, we had trouble integrating CNN with our prediction dataset and used our Logistic Regression Model which had a high accuracy as well. We then used this to predict tourist attractions in Charlottesville the user would find interesting with our model and displayed in our Anvil App which integrated Google Collab where we wrote the Machine Learning code with scikit-learn and TensorFlow.

Link to application: 
