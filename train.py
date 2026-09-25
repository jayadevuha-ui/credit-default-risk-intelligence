from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

OUT=Path("outputs"); OUT.mkdir(exist_ok=True)
DATA_URL="https://raw.githubusercontent.com/AdityaPrithvinath/UCI_CreditCard_DefaultDataset/master/UCI_Credit_Card.csv"

df=pd.read_csv(DATA_URL).rename(columns={"default.payment.next.month":"default"}).drop(columns=["ID"],errors="ignore")
df["EDUCATION"]=df["EDUCATION"].replace({0:4,5:4,6:4})
df["MARRIAGE"]=df["MARRIAGE"].replace({0:3})

TARGET="default"
FEATURES=[c for c in df.columns if c!=TARGET]
CAT=["SEX","EDUCATION","MARRIAGE"]
NUM=[c for c in FEATURES if c not in CAT]
X=df[FEATURES]; y=df[TARGET].astype(int)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)

prep=ColumnTransformer([
    ("num",StandardScaler(),NUM),
    ("cat",OneHotEncoder(handle_unknown="ignore"),CAT)
])

models={
"LogisticRegression":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=42),
"RandomForest":RandomForestClassifier(n_estimators=350,min_samples_leaf=3,class_weight="balanced_subsample",n_jobs=-1,random_state=42),
"HistGradientBoosting":HistGradientBoostingClassifier(learning_rate=.06,max_iter=250,max_leaf_nodes=31,l2_regularization=1.0,random_state=42),
}
try:
    from xgboost import XGBClassifier
    models["XGBoost"]=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=.04,subsample=.85,colsample_bytree=.85,eval_metric="logloss",n_jobs=-1,random_state=42)
except Exception:
    pass

cv=StratifiedKFold(5,shuffle=True,random_state=42)
rows=[]; fitted={}
for name,clf in models.items():
    pipe=Pipeline([("prep",prep),("model",clf)])
    s=cross_validate(pipe,X_train,y_train,cv=cv,n_jobs=-1,scoring={"roc_auc":"roc_auc","pr_auc":"average_precision","f1":"f1"})
    pipe.fit(X_train,y_train)
    p=pipe.predict_proba(X_test)[:,1]; pred=(p>=.5).astype(int)
    rows.append({
        "model":name,
        "cv_roc_auc_mean":float(s["test_roc_auc"].mean()),
        "cv_roc_auc_std":float(s["test_roc_auc"].std()),
        "cv_pr_auc_mean":float(s["test_pr_auc"].mean()),
        "cv_f1_mean":float(s["test_f1"].mean()),
        "test_roc_auc":float(roc_auc_score(y_test,p)),
        "test_pr_auc":float(average_precision_score(y_test,p)),
        "test_f1_050":float(f1_score(y_test,pred))
    })
    fitted[name]=pipe

comparison=pd.DataFrame(rows).sort_values("cv_roc_auc_mean",ascending=False)
selected=comparison.iloc[0]["model"]; best=fitted[selected]

X_fit,X_val,y_fit,y_val=train_test_split(X_train,y_train,test_size=.2,stratify=y_train,random_state=123)
best.fit(X_fit,y_fit)
vp=best.predict_proba(X_val)[:,1]
ths=np.linspace(.10,.80,141)
threshold=float(ths[int(np.argmax([f1_score(y_val,(vp>=t).astype(int)) for t in ths]))])

best.fit(X_train,y_train)
p=best.predict_proba(X_test)[:,1]; pred=(p>=threshold).astype(int)
metrics={
"selected_model":selected,
"selection_rule":"highest mean 5-fold CV ROC-AUC on training split",
"dataset_rows":int(len(df)),
"default_rate":float(y.mean()),
"threshold":threshold,
"test_roc_auc":float(roc_auc_score(y_test,p)),
"test_pr_auc":float(average_precision_score(y_test,p)),
"test_accuracy":float(accuracy_score(y_test,pred)),
"test_precision":float(precision_score(y_test,pred,zero_division=0)),
"test_recall":float(recall_score(y_test,pred,zero_division=0)),
"test_f1":float(f1_score(y_test,pred,zero_division=0)),
"confusion_matrix":confusion_matrix(y_test,pred).tolist(),
"model_comparison":comparison.to_dict(orient="records")
}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
comparison.to_csv(OUT/"model_comparison.csv",index=False)
joblib.dump(best,OUT/"best_model.joblib")

for kind in ["roc","pr","cm"]:
    plt.figure(figsize=(7,5))
    if kind=="roc":
        RocCurveDisplay.from_predictions(y_test,p); plt.title("ROC Curve")
    elif kind=="pr":
        PrecisionRecallDisplay.from_predictions(y_test,p); plt.title("Precision-Recall Curve")
    else:
        ConfusionMatrixDisplay.from_predictions(y_test,pred); plt.title(f"Confusion Matrix @ {threshold:.3f}")
    plt.tight_layout(); plt.savefig(OUT/f"{kind}_curve.png" if kind!="cm" else OUT/"confusion_matrix.png",dpi=160); plt.close()

sample=np.random.RandomState(42).choice(len(X_test),size=min(3000,len(X_test)),replace=False)
pi=permutation_importance(best,X_test.iloc[sample],y_test.iloc[sample],n_repeats=5,random_state=42,n_jobs=-1,scoring="roc_auc")
fi=pd.DataFrame({"feature":FEATURES,"importance_mean":pi.importances_mean,"importance_std":pi.importances_std}).sort_values("importance_mean",ascending=False)
fi.to_csv(OUT/"feature_importance.csv",index=False)

plt.figure(figsize=(8,7))
top=fi.head(12).sort_values("importance_mean")
plt.barh(top["feature"],top["importance_mean"]); plt.xlabel("Decrease in ROC-AUC")
plt.title("Permutation Feature Importance"); plt.tight_layout()
plt.savefig(OUT/"feature_importance.png",dpi=160); plt.close()

print(json.dumps(metrics,indent=2))

# Reproducible build trigger
