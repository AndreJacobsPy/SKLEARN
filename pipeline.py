import dataclasses as dc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

@dc.dataclass
class Model:
    """Base class to perform basic data cleaning after importing data"""

    df: pd.DataFrame

    def drop_column(self, column_name: list[str]) -> None:
        self.df = self.df.drop(axis='columns', labels=column_name)
        return None

    def drop_duplicates(self):
        self.df = self.df.drop_duplicates()
        return None

    def drop_null(self) -> None:
        self.df = self.df.apply(lambda x: x.replace("", "NaN"))
        self.df = self.df.dropna()

        return None

    def data_split(self, target) -> None:
        self.train_ = self.df.sample(frac=0.66, random_state=42)
        self.test_ = self.df.drop(self.train_.index)

        self.train_target = self.train_.pop(target)
        self.test_target = self.test_.pop(target)

        return None


@dc.dataclass
class Regressor(Model):
    def train(self, algorithm=LinearRegression) -> None:
        self.model = algorithm()
        self.model.fit(self.train_.values, self.train_target.values.reshape(-1, 1))

        return None

    def test(self) -> None:
        self.score = self.model.score(self.test_.values, self.test_target.values)
        self.predictions = self.model.predict(self.test_.values)

        return None

    def plot(self) -> plt.figure:
        fig, ax = plt.subplots()

        ax.set(
            title='Actual vs Predicted',
            xlabel=self.test_.columns[0],
            ylabel=self.test_target.name
        )
        ax.scatter(
            x=self.test_, 
            y=self.test_target
        )
        ax.plot(
            self.test_,
            self.predictions, 
            label='prediction', c='orange'
        )
        plt.legend()

        return fig

@dc.dataclass
class Classifier(Regressor):
    def train(self, algorithm=GaussianNB) -> None:
        self.model = algorithm()
        self.model.fit(self.train_.values, self.train_target.values.reshape(-1, 1))

        return None

    def confusion_matrix(self) -> plt.figure:
        cm = confusion_matrix(self.test_target, self.predictions)

        fig, ax = plt.subplots()
        
        ax.set_xticks(np.arange(len(self.test_target.unique())), self.test_target.unique())
        ax.set_yticks(np.arange(len(self.test_target.unique())), self.test_target.unique())
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')

        for row in range(cm.shape[0]):
            for column in range(cm.shape[1]):
                ax.text(
                    x=row, y=column, s=cm[row, column], 
                    va='center', ha='center', size='x-large')
        
        ax.imshow(cm)

        return fig
        


if __name__ == "__main__":
    data = load_iris()
    target = data.target
    inputs = data.data
    
    df = pd.DataFrame(inputs)
    df.columns = data.feature_names
    df['target'] = target
    df.to_csv('iris.csv')

    print(df.head())

    my_model = Classifier(df)
    my_model.data_split('target')
    my_model.train()
    my_model.test()

    print(my_model.score)
    print(data.target_names)