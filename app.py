from pathlib import Path
import json, joblib, pandas as pd, streamlit as st

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"outputs"/"best_model.joblib"
METRICS=ROOT/"outputs"/"metrics.json"

st.set_page_config(page_title="Credit Default Risk Intelligence",page_icon="📊",layout="wide")
st.title("Credit Default Risk Intelligence")
st.caption("Educational portfolio demo — not a real lending decision system.")

if not MODEL.exists():
    st.warning("Model artifact not found yet. Run python train.py or wait for the GitHub Actions build.")
    st.stop()

model=joblib.load(MODEL)
m=json.loads(METRICS.read_text())
threshold=float(m["threshold"])

with st.sidebar:
    st.metric("Test ROC-AUC",f"{m['test_roc_auc']:.3f}")
    st.metric("Test PR-AUC",f"{m['test_pr_auc']:.3f}")
    st.metric("Threshold",f"{threshold:.3f}")

with st.form("form"):
    c1,c2,c3=st.columns(3)
    with c1:
        limit_bal=st.number_input("Credit limit (NT$)",10000,1000000,150000,10000)
        age=st.number_input("Age",18,100,35)
    with c2:
        sex=st.selectbox("Sex code",[1,2])
        education=st.selectbox("Education",[1,2,3,4])
    with c3:
        marriage=st.selectbox("Marital status",[1,2,3])

    st.subheader("Repayment status")
    pay_names=["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    cols=st.columns(6); pays={}
    for col,name in zip(cols,pay_names):
        with col: pays[name]=st.slider(name,-2,8,0)

    st.subheader("Bill amounts")
    bills={}
    cols=st.columns(3)
    for i,name in enumerate([f"BILL_AMT{i}" for i in range(1,7)]):
        with cols[i%3]: bills[name]=st.number_input(name,-500000,2000000,50000,1000,key=name)

    st.subheader("Previous payment amounts")
    payamts={}
    cols=st.columns(3)
    for i,name in enumerate([f"PAY_AMT{i}" for i in range(1,7)]):
        with cols[i%3]: payamts[name]=st.number_input(name,0,2000000,5000,1000,key=name)

    go=st.form_submit_button("Estimate model score",use_container_width=True)

if go:
    row={"LIMIT_BAL":limit_bal,"SEX":sex,"EDUCATION":education,"MARRIAGE":marriage,"AGE":age,**pays,**bills,**payamts}
    prob=float(model.predict_proba(pd.DataFrame([row]))[0,1])
    st.metric("Estimated default probability",f"{prob:.1%}")
    st.write("Model classification:", "Default-risk class" if prob>=threshold else "Non-default-risk class")
    st.progress(float(max(0,min(1,prob))))
    st.info("This output is a model demonstration, not a credit recommendation.")
