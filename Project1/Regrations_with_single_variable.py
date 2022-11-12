


# importing the importante models.
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# reading the data from csv file as dataframe. 
data = pd.read_csv('ML.csv', header=None, names = ['Populations', 'Profits']) 

# inset new column with a value of number1 ---> allow will used it to find theta 0 and theta 1 later
data.insert(0, 'One', 1)                      # insert new columns called One at first index and have a value of 1

# sprate the first and second columns as X values and column 3 as y values.
number_of_columns = data.shape[1]             # get the number of columns in the data frame
X = data.iloc[:, 0 : number_of_columns-1]     # index column 1 and 2 and keep it as X
y = data.iloc[:, number_of_columns-1: number_of_columns] 


# convert the dataFrame into matrix to calculate the cost and theta easy.
X = np.matrix(X.values)                       # convert X columns into matrix.
y = np.matrix(y.values)                       # convert X columns into matrix.
theta = np.matrix(np.array([0,0]))            # inital values od from theta 0 and theta 1
# print(theta)


# calculating the cost j "Error" ---> the diff between the production and real values.
def cost_function(X,y,theta) :
    h = np.power(((X * theta.T) - y), 2)     # the productive value and power it to 2
    return np.sum(h) / (2 * len(X))

print(f'The cost j Value is {cost_function(X,y,theta)}')

""" 
The cost value is ---> 32.073 which is high, need to reduce it
through findint the perfic value for theta 0 and theta 1 that 
reduce it(using gradient Descent)
""" 

def gradient_descent(X, y, theta, alpha, iterations) :
    
    temp = np.matrix(np.zeros(theta.shape[1]))    # create a new matrix with 1 row and 2 columns to store theta values ---> 2D
    parameter = theta.ravel().shape[1]            # number if columns in the theta array.
    cost_j = np.matrix(np.zeros(iterations))      # a matrix with 1000 zeros. inside it will store the cost value
    
    for trys in range(iterations) :
        error = (X * theta.T) - y
        
        for j in range(parameter) :
            term = np.multiply(error, X[:, j]) 
            temp[0, j] = theta[0,j] - (( alpha / len(X) ) * np.sum(term))
        
        theta = temp 
        cost_j[0,trys] = cost_function(X, y, theta)
    return theta, cost_j 

alpha = 0.01                                   # learning rate
iterations = 1000                              # numbers of tries to find the perific values for theta 0 and theta1


new_theta, all_cost_tries = gradient_descent(X, y, theta, alpha, iterations) 
print(f'theta values are {new_theta}')
print(f'cost values {all_cost_tries}')
print(f'Cost j is {cost_function(X,y,new_theta)}')

# using linspace to divide the dataframe values.
data_values = np.linspace(data.Populations.min(), data.Profits.max(), 100)


# using linear equations to find hx value (productive values) 
productive_value = new_theta[0,0] + (new_theta[0,1] * data_values) 
print(productive_value)


#   drow the line. 
plt.style.use('dark_background')
figure, ax = plt.subplots(figsize =(5,5))
ax.plot(data_values, productive_value, 'r', label='Production')
ax.scatter(data.Populations, data.Profits, label='Training Data')
ax.legend(loc=2)
ax.set(xlabel='Populations', ylabel='Profits', title='Producted Profit vs Populations Size')
