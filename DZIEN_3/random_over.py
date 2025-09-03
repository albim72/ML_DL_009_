import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Tworzymy sztuczny zbiór: dużo szumu, mało sygnału
X, y = make_classification(
    n_samples=2000,
    n_features=50,
    n_informative=2,   # tylko 2 cechy faktycznie coś znaczą
    n_redundant=0,
    n_classes=2,
    flip_y=0.4,        # 40% etykiet losowo zamienione → chaos
    random_state=42
)

# Podział
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Mroczny, ciężki Random Forest – za dużo drzew, bez ograniczeń
rf_dark = RandomForestClassifier(
    n_estimators=500,   # ogromny las
    max_depth=None,     # drzewa idą na maxa
    random_state=42
)

# Trening
rf_dark.fit(X_train, y_train)

# Predykcja
y_train_pred = rf_dark.predict(X_train)
y_test_pred = rf_dark.predict(X_test)

# Wyniki
print("Wynik trening:", accuracy_score(y_train, y_train_pred))
print("Wynik test:", accuracy_score(y_test, y_test_pred))
