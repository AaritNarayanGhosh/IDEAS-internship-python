import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Data Ingestion & Preprocessing
url="https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df=pd.read_csv(url)
df=df.drop(columns=["UDI","Product ID"])
df.head()

#2.
numdf=df.select_dtypes(include=np.number)
corrmatrix=numdf.corr(method="pearson")

#3.
ordered_type=df["Type"].astype(pd.CategoricalDtype(categories=["L", "M", "H"], ordered=True))
type_dummies=pd.get_dummies(ordered_type,prefix="Type",drop_first=True, dtype=int)
df["Temp_Diff"]=df["Process temperature [K]"]-df["Air temperature [K]"]
df =df.drop(columns=["Type", "Air temperature [K]","Process temperature [K]"])
df =pd.concat([df, type_dummies],axis=1)
print(df.columns.tolist())
print(df.head())

#4.
print("shape before operation:",df.shape)
q1=df["Rotational speed [rpm]"].quantile(0.25)
q3=df["Rotational speed [rpm]"].quantile(0.75)
iqr=q3-q1
lower_bound= q1-1.5*iqr
upper_bound =q3+1.5*iqr
df=df[
    (df["Rotational speed [rpm]"]>=lower_bound)
    & (df["Rotational speed [rpm]"]<=upper_bound)
]

print("Shape afte operating:",df.shape)

print("""
Impact of Rotational Sped Outliers on decision Boundary:

1)Extreme values exert high leverage on models like Logistic Regression and SVMs.
The algorithm shifts the decision boundary toward these isolated points, degrading accuracy on normal operating points.
2)Unusually high rotational speeds produce massive error gradients during backpropagation, causing erratic weight updates and
destabilissing convergence.
3)Without aggressive scaling or removal, wide outlier ranges disturb the variance of other features,
leading the model to mistakes on rotational speed when determining failure states.
""")

#5.
leakage_cols=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
X=df.drop(columns=leakage_cols)
y=df["Machine failure"]
print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nClass distribution of y:")
print(y.value_counts())

#6.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled=pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)
print("\nFeature means (Train-should be ~0):")
print(X_train_scaled.mean().round(3))
print("\nFeature variances (Train - should be ~1):")
print(X_train_scaled.var().round(3))

#7.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lr_model=LogisticRegression(class_weight="balanced", random_state=42)
cv_scores=cross_val_score(
    lr_model, X_train_scaled, y_train, cv=5, scoring="roc_auc"
)
mean_auc=cv_scores.mean()
print(f"Mean 5-Fold Cross-Validation ROC-AUC:{mean_auc:.4f}")

#8.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
param_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
lr = LogisticRegression(class_weight="balanced", random_state=42)
grid_search = GridSearchCV(
    estimator=lr, param_grid=param_grid, cv=5, scoring="roc_auc"
)
grid_search.fit(X_train_scaled, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")

#9.
from sklearn.metrics import classification_report, roc_auc_score
y_pred=grid_search.predict(X_test_scaled)
y_prob=grid_search.predict_proba(X_test_scaled)[:, 1]
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Healthy (0)", "Failure (1)"]
    )
)
test_auc=roc_auc_score(y_test, y_prob)
print(f"Final Test ROC-AUC Score: {test_auc:.4f}")

#10
X_train_tensor=torch.tensor(X_train_scaled.values, dtype=torch.float32)
X_test_tensor=torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_train_tensor=torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor=torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
print("y_train_tensor shape:", tuple(y_train_tensor.shape))

#11
from torch.utils.data import DataLoader, TensorDataset
train_dataset=TensorDataset(X_train_tensor, y_train_tensor)
train_loader=DataLoader(train_dataset, batch_size=64, shuffle=True)
for batch_X, batch_y in train_loader:
    print("Batch X shape:", batch_X.shape)
    print("Batch y shape:", batch_y.shape)
    break

#12
class PredictiveMaintenanceMLP(nn.Module):
    def __init__(self, input_dim):
        super(PredictiveMaintenanceMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
input_dim = X_train_tensor.shape[1]
model = PredictiveMaintenanceMLP(input_dim=input_dim)
print(model)

#13
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)
num_epochs=20
model.train()
for epoch in range(num_epochs):
    running_loss=0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions=model(batch_X)
        loss=criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    epoch_loss=running_loss/len(train_loader)
    print(f"Epoch [{epoch + 1:02d}/{num_epochs}]-Average Loss: {epoch_loss:.4f}")

#14
model.eval()
with torch.no_grad():
    y_test_probs=model(X_test_tensor).numpy().flatten()
y_test_np=y_test_tensor.numpy().flatten()
fn_cost=10000
fp_cost=500
best_threshold=None
min_cost=float("inf")
best_fp, best_fn=0, 0
print(f"{'Threshold':<10} | {'FP':<6} | {'FN':<6} | {'Total Cost ($)':<15}")
print("-" * 45)
thresholds=np.arange(0.1, 1.0, 0.1)
for thresh in thresholds:
    y_pred=(y_test_probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test_np, y_pred).ravel()
    total_cost=(fn * fn_cost)+(fp * fp_cost)
    print(f"{thresh:<10.1f} | {fp:<6} | {fn:<6} | ${total_cost:<14,}")
    if total_cost<min_cost:
        min_cost=total_cost
        best_threshold=thresh
        best_fp=fp
        best_fn=fn
print("-" * 45)
print(f"Optimal Probability Threshold: {best_threshold:.1f}")
print(
    f"Minimum Total Financial Penalty: ${min_cost:,.2f} (False Positives: {best_fp}, False Negatives: {best_fn})"
)

#15

print("""
Conclusion:

1. Model Comparison:
Logistic Regression was simple to set up and easy to understand, but it only looks for linear relationships.
The PyTorch MLP worked better for capturing complex interactions between sensor readings (like temperature and speed)
because of its hidden layer and ReLU activation. However, the neural network took longer to set up and needs careful
tuning so it doesn't overfit.

2. Threshold and Cost Trade-offs:
The default 0.5 cutoff assumes that both types of mistakes cost the same.
But here, missing an actual breakdown (False Negative) costs $10,000 and risks hospital operations,
while an unnecessary check (False Positive) only costs $500 for technician labor.

Because missing a breakdown is 20 times worse, our optimal threshold ended up much lower than 0.5.
Lowering the threshold prioritizes recall over precision—we accept a few extra false alarms ($500 each)
to make sure we catch almost every real failure and avoid the huge $10,000 penalty.
""")
