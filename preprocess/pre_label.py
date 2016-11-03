import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

label_chunks = pd.read_csv("/home/jason/datamining/datasets/train_label.csv")
#label_chunks = label_chunks.head(10)

label_chunks["user_id"] = 0
label_chunks["merchant_id"] = 0

for i in range(len(label_chunks)):
	s = label_chunks.iloc[i,0].split("#")
	label_chunks.iloc[i,2] = s[0]
	label_chunks.iloc[i,3] = s[1]

label_chunks = label_chunks.drop('user_id#merchant_id',axis = 1)

label_chunks.to_csv("/home/jason/datamining/data/train_label.csv", index = False)
