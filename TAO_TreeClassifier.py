# TAO_TreeClassifier.py

from sklearn.ensemble import RandomForestClassifier


class TAO_TreeClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)
