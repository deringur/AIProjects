import pandas as pd   # pip install pandas 
import os # Good for navigating your computer's files 
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np   

# - pip install scikit-learn
# - pip install sklearn
# - pip install scipy
# Note: Even after installing the packages above, I continued to get the error: "liblapack.3.dylib' (no such file)"
# Only after doing the force-reinstall on Macbook (M1) it worked (see https://developer.apple.com/forums/thread/693696)
# - pip install --upgrade --force-reinstall scikit-learn
# Note: 
# 1) I had to install gfortran to install scipy, scikit-learn
# https://github.com/fxcoudert/gfortran-for-macOS/releases
# 2) Install openBlas "brew install openblas"

# Quiet deprecation warnings
import warnings
warnings.filterwarnings("ignore")


# Code from https://www.datacamp.com/tutorial/python-subprocess
def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def linearReg(car_data):
    X = car_data[['Age', 'Kms_Driven']]
    y = car_data[['Selling_Price']]
    
    fig, axs = plt.subplots(ncols=2, figsize = (8, 7))
    for ax in fig.get_axes():
        ax.label_outer()
    
    linear = LinearRegression()
    # train the model
    linear.fit(X[['Age']], y)
    

    y_pred = linear.predict(X[['Age']])

    sns.scatterplot(x = 'Age', y = 'Selling_Price', data = car_data, ax=axs[0])
    plt.xlabel('Age') # set the labels of the x and y axes
    plt.ylabel('Selling_Price (lakhs)')
    axs[0].plot(X['Age'], y_pred, color='red')

    linear.fit(X[['Kms_Driven']], y)
    y_pred = linear.predict(X[['Kms_Driven']])

    sns.scatterplot(x = 'Kms_Driven', y = 'Selling_Price', data = car_data, ax=axs[1])
    plt.xlabel('Kms_Driven') # set the labels of the x and y axes
    plt.ylabel('Selling_Price (lakhs)')
    axs[1].plot(X['Kms_Driven'], y_pred, color='red')

def linearReg3D(car_data):
# Seaborn 3D plot example is inspired from https://www.educba.com/seaborn-3d-plot/
    plt.figure (figsize = (8, 7))
    seaborn_plot = plt.axes (projection='3d')
    seaborn_plot.scatter3D(car_data['Age'], car_data['Kms_Driven']/1000, car_data['Selling_Price'])
    seaborn_plot.set_xlabel ('Age')
    seaborn_plot.set_ylabel ('Kms (x1000)')


def main():
    if not os.path.isfile('car_dekho.csv'):
        runcmd('/opt/homebrew/bin/wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"', verbose = True)

    data_path  = 'car_dekho.csv'
    car_data = pd.read_csv(data_path)

    linearReg(car_data)

    linearReg3D(car_data)

    plt.show()


if __name__ == "__main__":
    main()

