import numpy as np
import pandas as pd

class LinearRegressionModel:
    def __init__(self, learning_rate=0.05, epochs=10000, reg_lambda=0.001, normalize=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        self.normalize = normalize
        self.weights = None
        self.means = None
        self.stds = None
        self.loss_history = []  # պահելու ենք սխալների պատմությունը

    # ===================== FIT =======================
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)

        # ⚙️ 1️⃣ Ստանդարտացում (Normalization)
        if self.normalize:
            self.means = X.mean(axis=0)
            self.stds = X.std(axis=0)
            self.stds[self.stds == 0] = 1  # զրոյական std-ից խուսափելու համար
            X = (X - self.means) / self.stds

        # bias column ավելացում
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        # սկզբնական կշիռներ
        self.weights = np.random.randn(X.shape[1], 1) * 0.01

        # գրադիենտային վայրեջք
        for i in range(self.epochs):
            y_pred = X @ self.weights
            error = y_pred - y

            # 3️⃣ կանոնավորում (regularization)
            grad = (2 / len(y)) * (X.T @ error) + 2 * self.reg_lambda * self.weights
            grad[0] -= 2 * self.reg_lambda * self.weights[0]  # bias-ի վրա չկիրառել

            # կշիռների թարմացում
            self.weights -= self.learning_rate * grad

            # պահում ենք սխալի պատմությունը
            mse = np.mean(error ** 2)
            self.loss_history.append(mse)

            if i % 2000 == 0:
                print(f"Էպոխ {i}: MSE = {mse:.2f}")

        print("\nՈւսուցումն ավարտվեց ✅")

    # ===================== PREDICT =======================
    def predict(self, X):
        X = np.array(X, dtype=float)
        if self.normalize and self.means is not None:
            X = (X - self.means) / self.stds
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
        return X @ self.weights

    # ===================== EVALUATE =======================
    def evaluate(self, X, y):
        y = np.asarray(y, float).reshape(-1, 1)
        y_pred = self.predict(X)
    
        mse = float(np.mean((y_pred - y)**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_pred - y)))
    
        # 👉 Հաշվում ենք միջին տոկոսային սխալը
        mean_price = float(np.mean(y))
        percent_error = (mae / mean_price) * 100 if mean_price != 0 else 0

        ss_res = float(np.sum((y - y_pred)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    
        print(f"MSE = {mse:.2f}, RMSE = {rmse:.2f}, MAE = {mae:.2f} դրամ ({percent_error:.2f}%)")
        print(f"R² = {r2:.3f}")
        return y_pred

    

# Կարդում ենք houses.csv ֆայլը
df = pd.read_csv("houses.csv")

# Շենքի վիճակ (կարգային արժեքներ)
condition_map = {
    "Վատ": 0,
    "Վերանորոգված": 1,
    "Նորակառույց": 2
}
df["Condition"] = df["Condition"].map(condition_map)

# Թաղամասեր (կատեգորիալ արժեքներ՝ առանց կարգի)
district_map = {
    "Կենտրոն": 4,
    "Արաբկիր": 3,
    "Աջափնյակ": 2,
    "Կոմիտաս": 1,
    "Շենգավիթ": 0,
    "Նոր Նորք": 5
}
df["District"] = df["District"].map(district_map)

# # Ստուգենք արդյունքը
# print(df.head(10))

# Train/Test բաժանում
train = df.iloc[:20]
test = df.iloc[20:]

# # Train և Test հավաքածուների չափերը
# print("Train հավաքածուի չափը:", train.shape)
# print("Test հավաքածուի չափը:", test.shape)

# Փոխակերպում X և y մասերի
X_train = train[["Size", "District", "Condition", "Rooms"]].values
y_train = train["Price"].values.reshape(-1, 1)

X_test = test[["Size", "District", "Condition", "Rooms"]].values
y_test = test["Price"].values.reshape(-1, 1)


# Կառուցում ենք և ուսուցանում
model = LinearRegressionModel(learning_rate=0.05, epochs=10000)
model.fit(X_train, y_train)

# Գնահատում ենք test տվյալների վրա
y_pred = model.evaluate(X_test, y_test)

# Կանխատեսում նոր տան արժեքի համար
new_house = [[90, 3, 2, 3]]  # [Size, District, Condition, Rooms]
predicted_price = model.predict(new_house)
print(f"Նոր տան կանխատեսված գինը: {int(predicted_price[0][0])}$")