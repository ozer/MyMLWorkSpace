import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Market_Basket_Optimisation.csv",
                      header=None)

transactions = []

for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

from Apriori_Python.apyori import apriori

# min support if we want at least the number of the sale of 3 three times a day, then it makes 21 to a week
# why consider week ? Well, the dataset contains transactions through a week.
# min support calculation  3*7 / 7501 = 0,003

# min confidence means the required validation rate
#  a bit lower confidence level returns some rules
#  otherwise, higher confidence level may return a few rule and they can even be non-sense in the way of associate
#  20 % percent, 0.20, is a good combination with min_support which is 0.003

# the lift
#  we set it to three so that we can acquire some relevant rules
# ofc, it depends on the business rules

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.20, min_lift=3, min_length=2)

# Visualizing the rules

results = list(rules)
