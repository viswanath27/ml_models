import numpy as np 
import pandas as pd 

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def data_creation():
	print("")
	print("--------------Load Data from the Dataset --------------------")
	#Load the complete dataset 
	boston_dataset = load_boston()
	#Create the feature list 
	X_features = boston_dataset.data
	#Create the target list 
	Y_target = boston_dataset.target
	print ("X_features.shape - {},Y_target.shape - {}".format(X_features.shape,Y_target.shape))
	return X_features,Y_target


#This is more of the Test and Train split
def split_train_test(X_features,Y_target,train_test_ratio):
	print("")
	print("--------------Split Train and Test Data---------------------")
	#now spearate the train and test data set 
	X_train, X_test, Y_train, Y_test = train_test_split(X_features,Y_target, test_size=train_test_ratio, random_state=0)
	print ("X_train.shape - {}, X_test.shape - {}, Y_train.shape - {}, Y_test.shape - {}".format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))
	return X_train, X_test, Y_train, Y_test

def model_creation(X_train, Y_train):
	print("")
	#Instantiate the model 
	print("--------------Model Create---------------------")
	linereg = LinearRegression()
	linereg.fit(X_train,Y_train)
	# This is the Y intercept whcih will be used for the zero value of the x 
	print("the estimated intercept :{}".format(linereg.intercept_))
	
	#This will print the coiffcient values based on the each feature
	print("The coifficent is :{}".format(linereg.coef_))
	
	#Print the length of the coifficient values
	print("The coifficent is :{}".format(len(linereg.coef_)))
	return linereg

def results_analysis(linereg,X_test,Y_test):
	print("")
	print("--------------Result Summary---------------------")
	#this will give the MSE for the model
	print("MSE vlaue is :{}".format(np.mean((linereg.predict(X_test)-Y_test)**2)))
	
	#Calulate the model accuracy using the variance 
	print ("Variance score :{}".format(linereg.score(X_test,Y_test)))

def conclusion_summary():
	print("")
	print("--------------Result Summary---------------------")


#This is the core code of the ML steps which are performed 
def main():
	#This is basically create the required data from the dataset
	X_features,Y_target = data_creation()
	
	#This is used to split the data
	X_train, X_test, Y_train, Y_test = split_train_test(X_features,Y_target,0.3)	
	
	#This will return the trained model with the trian data 
	linereg = model_creation(X_train, Y_train)
	
	#This will use the test data for showing results 
	results_analysis(linereg,X_test,Y_test)
	
	#Conclusion summary
	conclusion_summary()

main()