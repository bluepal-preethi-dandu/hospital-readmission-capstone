import unittest

import pandas as pd

from src.train_pipeline import make_pipeline


class TrainPipelineTests(unittest.TestCase):
    def test_make_pipeline_accepts_feature_frame(self):
        X = pd.DataFrame(
            {
                "age": [30, 45, 60, 75],
                "num_medications": [5, 8, 12, 10],
                "race": ["Caucasian", "AfricanAmerican", "Asian", "Hispanic"],
            }
        )
        y = pd.Series([0, 1, 0, 1], name="readmitted_binary")

        model = make_pipeline("logistic_regression", X)
        model.fit(X, y)

        self.assertTrue(hasattr(model, "predict"))


if __name__ == "__main__":
    unittest.main()
