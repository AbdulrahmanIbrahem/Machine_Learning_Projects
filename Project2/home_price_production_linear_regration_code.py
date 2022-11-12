

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read dataset from csv file
data = pd.read_csv('home_size_price_number_of_rooms_dataset.csv', header=None, names=['Size', 'Bedrooms', 'Price']) 


""" 
rescalling data ---> due to the hug differance in rance between home Size, number of bedrooms 
and home price, need to rescall the data becouse will make the process diffecult
"""
data_after_rescalling = (data - data.mean()) / data.std()

# insert new colunm to the dataframe ---> column name one index 9 and value 1.
data_after_rescalling.insert(0, 'one', 1)

# sprate X and y values..
number_of_columns = data_after_rescalling.shape[1]                       # get the number of columns in the data fram
X = data_after_rescalling.iloc[:, 0 : number_of_columns-1]               # get the x values from the dataframe
y = data_after_rescalling.iloc[:, number_of_columns-1:number_of_columns] # get the y values from the dataframe


# convert the X, y into matrix and find the create theta.
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))

def compute_cost_function(X, y, theta) :
    
    error = np.power(((X * theta.T) - y), 2)                                   # calculate error
    cost_j = np.sum(error) / (2 * len(X))                                      # cost j
    return cost_j                                                              # return cost j value

print('The Cost function value is ' + str(compute_cost_function(X, y, theta))) 

# need to find the values for theta 0, theta 1 and theta 2
# using gradient desicent.
def gradinte_decinet(X, y, theta, alpha, iterations) :
    
    temp = np.matrix(np.zeros(theta.shape))                       # create a new matrix to store the new values of z inside them
    parameter = theta.ravel().shape[1]
    cost_j    = np.zeros(iterations)
    
    for iters in range(iterations) :
        error_value = (X * theta.T) - y
        
        for i in range(parameter) :
            term = np.multiply(error_value, X[:, i])
            temp[0,i] = theta[0,i] - ( (alpha / len(X)) * np.sum(term) )
        
        theta = temp
        cost_j[iters] = compute_cost_function(X, y, theta)
        
    return theta, cost_j
                     
alpha = 0.02
iterations = 1000


theta_values, cost_values = gradinte_decinet(X, y, theta, alpha, iterations)
print(theta_values)

print('The Cost function value is ' + str(compute_cost_function(X, y, theta_values))) 

x = np.linspace(data_after_rescalling.Size.min(), data_after_rescalling.Size.max(), 100) 
productive_value = theta_values[0,0] + (theta_values[0,1] * x)

figure, ax = plt.subplots(figsize=(5,6))
ax.plot(x, productive_value, 'red', label='Production')
ax.scatter(data_after_rescalling.Size, data_after_rescalling.Price, label='Training Data')
ax.legend(loc=2)
ax.set(xlabel='Size', ylabel='Price', title='Size vs. Price')

# dorwing error graph.
fig, ax = plt.subplots(figsize=(5,6))
ax.plot( cost_values, 'green')
ax.set(xlabel='Number of Tries',ylabel='Error', title='Error vs. Training Epoch')
print(cost_values)