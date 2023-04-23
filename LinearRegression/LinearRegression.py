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

def main():
    #if not os.path.isfile('car_dekho.csv'):
    runcmd('/opt/homebrew/bin/wget -q --show-progress "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%201%20-%205/Session%202a%20-%20Linear%20Regression/car_dekho.csv"', verbose = True)

    data_path  = 'car_dekho.csv'
    car_data = pd.read_csv(data_path)

    car_data.head() 
    print(len(car_data))

    sns.scatterplot(x = 'Age', y = 'Selling_Price', data = car_data)
    sns.catplot(x = 'Fuel_Type', y = 'Selling_Price', data = car_data, kind = 'swarm', s = 2)
    sns.catplot(x = 'Fuel_Type', y = 'Selling_Price', data = car_data, kind = 'box')
    sns.scatterplot(x = 'Kms_Driven', y = 'Selling_Price', data = car_data)
    X = car_data[['Age']]
    y = car_data[['Selling_Price']]


    plt.show()


if __name__ == "__main__":
    main()

