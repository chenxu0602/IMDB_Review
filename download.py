import os, sys
import numpy as np
import pandas as pd
import pyprind
from pathlib import Path

basepath = Path("aclImdb")

labels = {"pos": 1, "neg": 0}
pbar = pyprind.ProgBar(50000, stream=sys.stdout)

reviews, sentiments = [], []

for s in ("test", "train"):
    for l in ("pos", "neg"):
        path = basepath / s / l
        for file in sorted(path.iterdir()):
            with open(file, 'r', encoding="utf-8") as infile:
                txt = infile.read()

            reviews.append(txt)
            sentiments.append(labels[l])
            pbar.update()

df = pd.DataFrame({
    "review": reviews,
    "sentiment": sentiments,
})

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv("movie_data.csv", index=False, encoding="utf-8")