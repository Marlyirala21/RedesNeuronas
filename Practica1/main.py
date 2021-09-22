import os
import pandas as pd

# leemos y guardamos los datos
df = pd.read_csv('dataset.csv', header=0)

# normalizamos los datos con el método Min-max de cada columna del dataFrame
for col in df:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Utilizando la libreria de pandas hacemos la aleatorización al mismo tiempo que la separación
training = df.sample(frac=0.7) # 70% de los datos aleatorios
validation = df.sample(frac=0.15) # 15% de los datos aleatorios
test = df.sample(frac=0.15) # 15% de los datos aleatorios

# creamos los archivos correspondientes
training.to_csv("training.csv")
validation.to_csv("validation.csv")
test.to_csv("test.csv")


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    print("")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
