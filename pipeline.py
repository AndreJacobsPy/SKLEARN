import dataclasses as dc
import pandas as pd

from sklearn.linear_model import LinearRegression

@dc.dataclass
class Model(BaseException):
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
        self.train_ = self.df.sample(frac=0.66)
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


if __name__ == "__main__":
    df = pd.read_csv('RSM_event.csv')
    my_model = Regressor(df)
    my_model.drop_null()
    my_model.data_split('score')
    my_model.train()
    # my_model.test()
    # my_model.plot_fit()

    print(my_model.df)