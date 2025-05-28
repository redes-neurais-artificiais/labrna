
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("""\n\n=====================================================
Carregando o dataset...
-----------------------------------------------------""")
data = fetch_openml(name='wine-quality-red', version=1, as_frame=True)
print("ok!")
X = data.data.values
y = data.target.astype(int).values
y = (y >= 6).astype(int)  # binarização

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
