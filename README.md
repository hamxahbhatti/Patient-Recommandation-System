 # Patient Recommandation System
This repo contains a model trained on timeseries dataset of number of patient of different diseases who visited a hospital throughout the year. This model can predict total number of the patients of different diseases that can visit hospital next month.

# Usage 
 recommadationsystem.py contains the script that is used to create the model . i have used ANNs instead of RNN because i have only 100 observation and RNNs requires a lot more.
 main.py will excute the code that will load the trained model and then predict the next month patient's list
