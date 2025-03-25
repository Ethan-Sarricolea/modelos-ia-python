import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class WeightIdentifier:

    def __init__(self):
        self.document = pd.read_csv("assets\pesos.csv")
        
        self.x_process = self.document["altura"].values.reshape(-1,1)
        self.y_process = self.document["peso"].values.reshape(-1,1)

        self.model = LinearRegression()

        self.model.fit(self.x_process, self.y_process)

    def showData(self):
        plt.scatter(x="altura",
                    y="peso",
                   data=self.document)
        plt.show()
        
    def getScore(self):
        return self.model.score(self.x_process, self.y_process)
        
    def ask(self):
        print("Mi porcentaje de atine es de: ", self.getScore())
        print("Coloca tu altura (cm) y predecire tu peso: ")
        try:
            altura = int(input("altura: "))
            prediccion = self.model.predict([[altura]])
            print("Tu peso deberia ser de: ",prediccion)
        except Exception as e:
            print("Ha ocurrido un error: ", e)