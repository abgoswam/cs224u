import numpy as np
import pandas as pd
import vsm

raw_data = {
    'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons',
                 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
    'company': ['infantry', 'infantry', 'cavalry', 'cavalry', 'infantry', 'infantry', 'cavalry', 'cavalry', 'infantry',
                'infantry', 'cavalry', 'cavalry'],
    'experience': ['veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie', 'veteran', 'rookie', 'veteran',
                   'rookie', 'veteran', 'rookie'],
    'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani',
             'Ali'],
    'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
    'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}

df = pd.DataFrame(raw_data,
                  columns=['regiment', 'company', 'experience', 'name', 'preTestScore', 'postTestScore'])

print(df)

gnarly_df = pd.DataFrame(
    np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1]], dtype='float64'),
    index=['gnarly', 'wicked', 'awesome', 'lame', 'terrible'])

gnarly_df

vsm.neighbors('gnarly', gnarly_df)

from mittens import GloVe
import numpy as np

cooccurrence = np.array([
    [4., 4., 2., 0.],
    [5., 61., 8., 18.],
    [2., 8., 10., 0.],
    [0., 18., 0., 5.]])

glove_model = GloVe(n=2, max_iter=100)
embeddings = glove_model.fit(cooccurrence)

embeddings
