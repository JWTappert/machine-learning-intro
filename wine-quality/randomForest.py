import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# Get the red wine data
red_wine_train = pd.read_csv('wine-quality-red-train.csv', delimiter=';')

# Get the white wine data
white_wine_train = pd.read_csv('wine-quality-white-train.csv', delimiter=';')

# add class column to red wine train
red_wine_train['class'] = 'red'

print red_wine_train