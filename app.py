import os
import urllib.request
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from ucimlrepo import fetch_ucirepo
import gradio as gr

# ==========================================
# 0. Automatic Dataset File Setup
# ==========================================
CLEVELAND   = 'processed.cleveland.data'  
HUNGARY     = 'hungarian.data'            
SWITZERLAND = 'switzerland.data'
LONG_BEACH  = 'long-beach-va.data'

base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
files_to_download = {
    CLEVELAND: base_url + "processed.cleveland.data",
    HUNGARY: base_url + "hungarian.data",
    SWITZERLAND: base_url + "switzerland.data",
    LONG_BEACH: base_url + "long-beach-va.data"
}

for filename, url in files_to_download.items():
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"Could not download {filename}: {e}")

heart_disease = fetch_ucirepo(id=45)
X_base = heart_disease.data.features
y_base = heart_disease.data.targets


# ==========================================
# 1. Cleveland Base Model & Top 5 Measurements
# ==========================================
print("==========================================")
print("1. CLEVELAND BASE MODEL & FEATURE IMPORTANCE")
print("==========================================")

df_base = X_base.copy()
df_base['num'] = y_base['num']
df_base['target'] = df_base['num'].apply(lambda x: 0 if x == 0 else 1)
df_base = df_base.drop(columns=['num'])

df_base['ca'] = df_base['ca'].fillna(df_base['ca'].mode()[0])
df_base['thal'] = df_base['thal'].fillna(df_base['thal'].mode()[0])

X_clean = df_base.drop(columns=['target'])
y_clean = df_base['target']

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

model_base = LogisticRegression(max_iter=1000)
model_base.fit(X_train_b, y_train_b)
y_pred_b = model_base.predict(X_test_b)

baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_b, y_train_b)
baseline_pred = baseline.predict(X_test_b)

acc_val = accuracy_score(y_test_b, y_pred_b)
rec_val = recall_score(y_test_b, y_pred_b)
base_acc_val = accuracy_score(y_test_b, baseline_pred)

print(f"Accuracy: {acc_val:.2f}")
print(f"Recall: {rec_val:.2f}")
print(f"Baseline Accuracy: {base_acc_val:.2f}")

coefficients = pd.DataFrame({
    'Measurement': X_clean.columns,
    'Coefficient': model_base.coef_[0]
})
coefficients['Abs_Coefficient'] = np.abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='Abs_Coefficient', ascending=False)

top5_df = coefficients[['Measurement', 'Coefficient']].head(5)
print("\n--- Top 5 Most Important Measurements ---")
print(top5_df)


# ==========================================
# 2. Multi-Hospital Domain Shift Evaluation
# ==========================================
print("\n==========================================")
print("2. MULTI-HOSPITAL DOMAIN SHIFT EVALUATION")
print("==========================================")

cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','num']
idx = [2,3,8,9,11,15,18,31,37,39,40,43,50,57]

if not os.path.exists(CLEVELAND):
    df_temp = X_base.copy()
    df_temp['num'] = y_base['num']
    df_temp.to_csv(CLEVELAND, header=False, index=False)

def finish(df):
    df = df.apply(pd.to_numeric, errors='coerce').replace(-9, np.nan)
    for c in ['chol', 'trestbps']:      
        df.loc[df[c] == 0, c] = np.nan
    df = df.dropna(subset=['num'])
    df['num'] = (df['num'] > 0).astype(int)
    return df.reset_index(drop=True)

def read_raw(path):
    rows, cur = [], []
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    for t in open(path).read().split():
        if t == 'name':
            if len(cur) == 75: 
                rows.append(cur)
            cur = []
        else:
            cur.append(t)
    if not rows:
        return pd.DataFrame(columns=cols)
    return finish(pd.DataFrame([[r[i] for i in idx] for r in rows], columns=cols))

cle = finish(pd.read_csv(CLEVELAND, header=None, names=cols, na_values='?'))
hun = read_raw(HUNGARY)
swi = read_raw(SWITZERLAND)
lbv = read_raw(LONG_BEACH)

cle_str = f"Cleveland   : {cle.shape} | Patients {cle['num'].mean():.2f}"
hun_str = f"Hungary     : {hun.shape} | Patients {hun['num'].mean():.2f}" if len(hun) > 0 else "Hungary     : N/A"
swi_str = f"Switzerland : {swi.shape} | Patients {swi['num'].mean():.2f}" if len(swi) > 0 else "Switzerland : N/A"
lbv_str = f"Long Beach  : {lbv.shape} | Patients {lbv['num'].mean():.2f}" if len(lbv) > 0 else "Long Beach  : N/A"

print(cle_str)
print(hun_str)
print(swi_str)
print(lbv_str)

miss = hun.drop(columns='num').isna().mean()
feats = miss[miss < 0.5].index.tolist()
print('\nSelected features based on Hungary availability:', feats)

pipeline_eval = Pipeline([
    ('imp', SimpleImputer(strategy='median', add_indicator=True)),
    ('sc', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000)),
])

pipeline_eval.fit(cle[feats], cle['num'])

ext_results = []
print("\n--- External Hospital Generalization Results ---")
if len(hun) > 0:
    p_hun = pipeline_eval.predict_proba(hun[feats])[:, 1]
    auc_h, rec_h, prec_h = roc_auc_score(hun['num'], p_hun), recall_score(hun['num'], (p_hun >= 0.5).astype(int), zero_division=0), precision_score(hun['num'], (p_hun >= 0.5).astype(int), zero_division=0)
    res_h = f"Hungary (external)   : AUC={auc_h:.3f}  recall={rec_h:.3f}  precision={prec_h:.3f}"
    print(res_h)
    ext_results.append(res_h)

if len(swi) > 0:
    p_swi = pipeline_eval.predict_proba(swi[feats])[:, 1]
    auc_s, rec_s, prec_s = roc_auc_score(swi['num'], p_swi), recall_score(swi['num'], (p_swi >= 0.5).astype(int), zero_division=0), precision_score(swi['num'], (p_swi >= 0.5).astype(int), zero_division=0)
    res_s = f"Switzerland (external): AUC={auc_s:.3f}  recall={rec_s:.3f}  precision={prec_s:.3f}"
    print(res_s)
    ext_results.append(res_s)

if len(lbv) > 0:
    p_lbv = pipeline_eval.predict_proba(lbv[feats])[:, 1]
    auc_l, rec_l, prec_l = roc_auc_score(lbv['num'], p_lbv), recall_score(lbv['num'], (p_lbv >= 0.5).astype(int), zero_division=0), precision_score(lbv['num'], (p_lbv >= 0.5).astype(int), zero_division=0)
    res_l = f"Long Beach (external) : AUC={auc_l:.3f}  recall={rec_l:.3f}  precision={prec_l:.3f}"
    print(res_l)
    ext_results.append(res_l)


# ==========================================
# 3. Clinical Decision Support Gradio Web App
# ==========================================
FEATURES = list(X_base.columns)
NUM = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CAT = [c for c in FEATURES if c not in NUM]

df = X_base.copy()
df["target"] = (y_base["num"] > 0).astype(int)

for c in CAT:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else 0).astype(int)

Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["target"], test_size=0.2, stratify=df["target"], random_state=42)

pre = ColumnTransformer(
    [("num", Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUM),
     ("cat", Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown="ignore"))]), CAT)],
    sparse_threshold=0)

model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])
model.fit(Xtr, ytr)

coef_app = model.named_steps["clf"].coef_.ravel()
names_app = model.named_steps["pre"].get_feature_names_out()
owner = []
for n in names_app:
    kind, rest = n.split("__", 1)
    owner.append(rest if kind == "num" else next(f for f in CAT if rest.startswith(f + "_")))

Xt_train_app = np.asarray(model.named_steps["pre"].transform(Xtr))
mean_t_app = Xt_train_app.mean(axis=0)

M = dict(pipe=model, owner=owner, mean_t=mean_t_app, coef=coef_app)

GROUPS = {
    "bp": ["trestbps"],
    "arteries": ["chol", "ca"],
    "muscle": ["thalach", "oldpeak", "exang", "slope", "restecg", "thal"],
    "sugar": ["fbs"],
    "chest": ["cp"],
    "baseline": ["age", "sex"],
}
CLINICAL = ["bp", "arteries", "muscle", "sugar", "chest"]
COLORS = ["#3DAA6D", "#E8C33A", "#F08A24", "#D7263D"]
BAND_COL = ["#3DAA6D", "#F08A24", "#D7263D"]

def explain(M, row):
    x = pd.DataFrame([row])[FEATURES]
    p = float(M["pipe"].predict_proba(x)[0, 1])
    xt = np.asarray(M["pipe"].named_steps["pre"].transform(x)).ravel()
    contrib = M["coef"] * (xt - M["mean_t"])
    
    per_feat = {}
    for f, c in zip(M["owner"], contrib):
        per_feat[f] = per_feat.get(f, 0.0) + float(c)
        
    per_group = {g: sum(per_feat[f] for f in fs if f in per_feat) for g, fs in GROUPS.items()}
    return p, per_group

def level(c):
    return 0 if c < 0.15 else 1 if c < 0.5 else 2 if c < 1.0 else 3

def band(p):
    return 0 if p < 0.30 else 1 if p < 0.60 else 2

T = {
    "en": {
        "name": {"bp": "blood pressure", "arteries": "cholesterol and blood vessels",
                 "muscle": "cardiac stress response", "sugar": "blood sugar",
                 "chest": "chest pain pattern"},
        "chip": {"bp": "Blood pressure", "arteries": "Cholesterol", "muscle": "Heart muscle",
                 "sugar": "Blood sugar", "chest": "Chest pain"},
        "band": ["low risk", "moderate risk", "high risk"],
        "risk_title": "Clinical Risk Assessment Index",
        "open": "Clinical evaluation summary. Based on patient metrics, the model estimates a risk index of <b>{pct}%</b> ({band}).",
        "lead_high": "Primary clinical driver elevating the risk index is <b>{name}</b>:",
        "lead_low": "Although overall risk is low, the primary contributor is <b>{name}</b>:",
        "others": "Secondary contributing factors: {list}.",
        "alone": "All other clinical parameters remain within stable ranges.",
        "baseline": "Patient age and sex establish baseline demographic risk. Clinical management focuses on modifiable risk factors.",
        "calm": "Patient clinical profile is stable: no primary parameter significantly elevates cardiac risk.",
        "keep": "Maintain routine clinical monitoring and preventive care guidelines.",
        "next": "Recommended clinical management actions:",
        "doctor": "Clinical decision support tool for professional medical evaluation.",
        "disc": "This system serves as a clinical prototype and does not replace formal medical diagnosis.",
        "meta": {
            "bp": "Persistent elevated vascular resistance increases myocardial workload and left ventricular strain.",
            "arteries": "Hypercholesterolemia and vascular luminal narrowing restrict myocardial perfusion.",
            "muscle": "Stress test indicators reveal potential myocardial ischemia or impaired functional capacity.",
            "sugar": "Elevated glycemic metrics accelerate endothelial dysfunction and atherosclerosis.",
            "chest": "Symptom presentation pattern requires thorough cardiological evaluation.",
        },
        "advice": {
            "bp": ["Initiate antihypertensive protocol and dietary sodium restriction.",
                   "Schedule ambulatory blood pressure monitoring (ABPM).",
                   "Prescribe structured aerobic exercise program contingent on cardiac clearance."],
            "arteries": ["Order comprehensive lipid panel and evaluate statin therapy candidacy.",
                         "Implement therapeutic lifestyle changes and nutritional counseling.",
                         "Consider advanced vascular imaging or cardiology referral."],
            "muscle": ["Review 12-lead ECG and exercise stress test findings.",
                       "Evaluate need for echocardiography or coronary angiography.",
                       "Monitor closely for anginal symptoms during physical exertion."],
            "sugar": ["Order fasting plasma glucose and HbA1c laboratory workup.",
                      "Evaluate glycemic control and dietary carbohydrate management.",
                      "Screen for metabolic syndrome indicators."],
            "chest": ["Perform detailed clinical workup of anginal characteristics.",
                      "Evaluate for acute coronary syndrome if symptoms are acute.",
                      "Schedule formal cardiology consultation."],
        },
        "legend": "Colour: <span style='color:#3DAA6D;font-weight:600'>green</span> = normal, <span style='color:#E8C33A;font-weight:600'>yellow</span> = mild elevation, <span style='color:#F08A24;font-weight:600'>orange</span> = high risk, <span style='color:#D7263D;font-weight:600'>red</span> = critical.",
    }
}

CP_LABEL = {
    "en": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "no chest pain (asymptomatic)"}
}

def reading(g, row, lang):
    if g == "bp":
        v = int(row["trestbps"])
        i = 0 if v < 120 else 1 if v < 130 else 2 if v < 140 else 3
        s = ("normal", "slightly elevated", "high", "very high")[i]
        return f"resting blood pressure is recorded at {v} mmHg ({s})."
    if g == "arteries":
        v = int(row["chol"])
        i = 0 if v < 200 else 1 if v < 240 else 2
        s = ("desirable", "borderline high", "high")[i]
        out = f"total serum cholesterol is {v} mg/dl ({s})"
        ca = int(row["ca"])
        if ca >= 1:
            out += f", with {ca} major vessel(s) exhibiting fluoroscopic obstruction."
        else:
            out += "."
        return out
    if g == "muscle":
        bits = []
        age, hr = int(row["age"]), int(row["thalach"])
        pct = round(100 * hr / max(220 - age, 1))
        if pct < 85:
            bits.append(f"peak heart rate achieved is {pct}% of age-predicted maximum")
        if int(row["exang"]) == 1:
            bits.append("exercise-induced angina observed")
        if float(row["oldpeak"]) >= 1:
            bits.append(f"significant ST-segment depression of {row['oldpeak']} mV")
        if int(row["thal"]) in (6, 7):
            bits.append("abnormal thallium perfusion scan findings")
        if not bits:
            bits.append("stress test parameters within acceptable limits")
        return ", ".join(bits) + "."
    if g == "sugar":
        return "fasting blood glucose level exceeds 120 mg/dl."
    if g == "chest":
        lab = CP_LABEL[lang][int(row["cp"])]
        return f"reported chest pain classification: {lab}."
    return ""

def build_story(p, per_group, row, lang):
    t = T[lang]
    b = band(p)
    clin = {g: per_group[g] for g in CLINICAL}
    main = max(clin, key=clin.get)
    has_main = clin[main] >= 0.15
    parts = [t["open"].format(pct=round(p * 100), band=t["band"][b])]
    if has_main:
        lead = t["lead_high"] if b > 0 else t["lead_low"]
        parts.append(lead.format(name=t["name"][main]) + " " + reading(main, row, lang)
                     + " " + t["meta"][main])
        others = [g for g, v in sorted(clin.items(), key=lambda kv: -kv[1]) if g != main and v >= 0.15]
        parts.append(t["others"].format(list=", ".join(t["name"][g] for g in others)) if others else t["alone"])
        if per_group["baseline"] >= 0.3:
            parts.append(t["baseline"])
        adv = "".join(f"<li>{a}</li>" for a in t["advice"][main])
        parts.append(f"<b>{t['next']}</b><ul style='margin:6px 0'>{adv}</ul>")
    else:
        parts += [t["calm"], t["keep"]]
    parts += [t["doctor"], f"<i>{t['disc']}</i>"]
    direction = "ltr"
    tag = lambda x: "div" if "<ul" in x else "p"
    body = "".join(f"<{tag(x)} style='margin:8px 0;line-height:1.7'>{x}</{tag(x)}>" for x in parts)
    return (f"<div dir='{direction}' style='padding:14px 16px;border-radius:14px;background:#fff;"
            f"color:#1f2933;border:1px solid #e1e5ea;font-size:15px'>{body}</div>")

def risk_card(p, lang):
    t = T[lang]
    b = band(p)
    direction = "ltr"
    return (f"<div dir='{direction}' style='padding:12px 16px;border-radius:14px;background:#fff;color:#1f2933;"
            f"border:1px solid #e1e5ea'><div style='font-size:13px;color:#52606d'>{t['risk_title']}</div>"
            f"<div style='font-size:34px;font-weight:700;color:{BAND_COL[b]}'>{round(p*100)}% "
            f"<span style='font-size:16px'>({t['band'][b]})</span></div>"
            f"<div style='height:10px;border-radius:5px;background:#e8ecf0;margin-top:6px'>"
            f"<div style='height:10px;border-radius:5px;width:{round(p*100)}%;background:{BAND_COL[b]}'></div></div></div>")

def _star(cx, cy, r1, r2, n=8):
    pts = []
    for i in range(2 * n):
        r = r1 if i % 2 == 0 else r2
        a = math.pi * i / n - math.pi / 2
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    return " ".join(pts)

def _chip(x, y, text, color, star=False):
    w = max(80, len(text) * 7 + 38)
    label = ("★ " if star else "") + text
    return (f'<g><rect x="{x - w/2:.0f}" y="{y - 14}" width="{w:.0f}" height="28" rx="14" fill="#fff" '
            f'stroke="{color}" stroke-width="2.5"/><circle cx="{x - w/2 + 15:.0f}" cy="{y}" r="5.5" fill="{color}"/>'
            f'<text x="{x + 8:.0f}" y="{y + 4}" text-anchor="middle" font-size="12.5" font-weight="600" '
            f'fill="#1f2933" font-family="sans-serif">{label}</text></g>')

BODY = ("M165 170 C140 210 142 300 175 355 C200 395 235 425 268 420 C316 410 336 345 330 275 "
        "C325 205 295 168 260 162 C240 158 222 172 205 170 C190 168 178 165 165 170 Z")
ARCH = "M205 172 C198 110 205 55 250 48 C292 42 312 85 306 150"
VESSELS = [("M212 178 C218 240 234 330 254 410", 6), ("M222 172 C275 180 315 215 324 285", 5),
           ("M185 168 C148 190 136 262 152 335", 6), ("M226 230 C250 232 270 245 282 262", 3.5),
           ("M236 300 C256 300 272 312 284 328", 3.5), ("M148 250 C165 252 178 262 186 280", 3.5),
           ("M214 215 C195 225 185 240 182 262", 3.5)]

def render_svg(per_group, lang):
    t = T[lang]
    lv = {g: level(per_group[g]) for g in CLINICAL}
    col = {g: COLORS[lv[g]] for g in CLINICAL}
    tint = {g: [0, .25, .42, .55][lv[g]] for g in CLINICAL}
    main = max(CLINICAL, key=lambda g: per_group[g])
    if per_group[main] < 0.15:
        main = None

    chip = lambda g, x, y: _chip(x, y, t["chip"][g], col[g], star=(g == main))
    arteries_glow = "".join(f'<path d="{d}" stroke="{col["arteries"]}" stroke-width="{w + 8}" filter="url(#glow)" opacity=".9"/>' for d, w in VESSELS)
    arteries_dark = "".join(f'<path d="{d}" stroke="#3a0d10" stroke-opacity=".55" stroke-width="{w + 3}"/>' for d, w in VESSELS)
    arteries_col = "".join(f'<path d="{d}" stroke="{col["arteries"]}" stroke-width="{w}"/>' for d, w in VESSELS)
    
    svg = f"""<svg viewBox="0 0 420 470" xmlns="http://www.w3.org/2000/svg" role="img"
 aria-label="Anatomical heart coloured by risk drivers" style="width:100%;max-width:520px;display:block;margin:auto">
<defs>
 <radialGradient id="gBody" cx=".62" cy=".4" r=".75"><stop offset="0" stop-color="#f08a84"/><stop offset=".5" stop-color="#cf4a47"/><stop offset="1" stop-color="#7f1620"/></radialGradient>
 <linearGradient id="gRV" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e9a3a0"/><stop offset="1" stop-color="#b53a3d"/></linearGradient>
 <linearGradient id="gAorta" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ea6660"/><stop offset="1" stop-color="#9c242a"/></linearGradient>
 <linearGradient id="gVein" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#8c92d4"/><stop offset="1" stop-color="#4b4f94"/></linearGradient>
 <filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="6"/></filter>
 <filter id="soft" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="7"/></filter>
</defs>
<style>
@keyframes heartBeat {{
    0%, 100% {{ transform: scale(1); }}
    15% {{ transform: scale(1.025); }}
    30% {{ transform: scale(1); }}
    45% {{ transform: scale(1.015); }}
    60% {{ transform: scale(1); }}
}}
.live-heart {{
    transform-origin: 240px 290px;
    animation: heartBeat 2.2s ease-in-out infinite;
}}
</style>
<rect width="420" height="470" rx="20" fill="#f4f6f8"/>
<ellipse cx="240" cy="452" rx="95" ry="8" fill="#1f2933" opacity=".12"/>

<!-- Heart body wrapped with smooth heartbeat effect only -->
<g class="live-heart">
    <path d="{ARCH}" fill="none" stroke="{col['bp']}" stroke-width="42" stroke-linecap="round" filter="url(#glow)" opacity=".95"/>
    <rect x="147" y="62" width="28" height="125" rx="12" fill="url(#gVein)"/>
    <path d="{ARCH}" fill="none" stroke="url(#gAorta)" stroke-width="32" stroke-linecap="round"/>
    <path d="{ARCH}" fill="none" stroke="{col['bp']}" stroke-opacity="{[0, .45, .65, .8][lv['bp']]}" stroke-width="32" stroke-linecap="round"/>
    <g fill="none" stroke="#c8403f" stroke-width="9" stroke-linecap="round"><path d="M226 52 L222 28"/><path d="M246 47 L247 23"/><path d="M266 50 L273 28"/></g>
    <path d="M235 178 C235 135 240 100 266 82" fill="none" stroke="#7c82c4" stroke-width="28" stroke-linecap="round"/>
    <path d="M266 82 C286 72 306 72 330 80" fill="none" stroke="#7c82c4" stroke-width="17" stroke-linecap="round"/>

    <path d="{BODY}" fill="none" stroke="{col['muscle']}" stroke-width="14" filter="url(#glow)" opacity=".95"/>
    <path d="M120 200 C112 168 150 150 188 165 L196 252 C160 278 125 258 120 200 Z" fill="url(#gRV)"/>
    <path d="{BODY}" fill="url(#gBody)"/>
    <path d="M165 170 C140 210 142 300 175 355 C190 380 210 400 225 410 C220 340 212 250 207 172 C190 168 178 165 165 170 Z" fill="url(#gRV)" opacity=".55"/>
    <path d="M268 163 C290 148 318 158 322 186 C300 192 282 184 268 163 Z" fill="#c2453f"/>
    <path d="{BODY}" fill="{col['muscle']}" opacity="{tint['muscle']}"/>
    <path d="M210 172 C216 240 232 330 252 412" fill="none" stroke="#f1d9a0" stroke-width="10" stroke-linecap="round" opacity=".55"/>
    <ellipse cx="262" cy="235" rx="26" ry="72" transform="rotate(14 262 235)" fill="#fff" opacity=".22" filter="url(#soft)"/>

    <g fill="none" stroke-linecap="round">{arteries_glow}{arteries_dark}{arteries_col}</g>
</g>

<!-- Static chips and side icons (No blinking) -->
<path d="M62 30 C62 30 34 66 34 86 a28 28 0 0 0 56 0 C90 66 62 30 62 30 Z" fill="{col['sugar']}" stroke="#fff" stroke-width="3"/>
<ellipse cx="50" cy="84" rx="5" ry="10" fill="#fff" opacity=".45"/>
<polygon points="{_star(372, 235, 28, 14)}" fill="{col['chest']}" stroke="#fff" stroke-width="3"/>

<g stroke="#9aa5b1" stroke-width="1.5" stroke-dasharray="3 3"><line x1="190" y1="36" x2="209" y2="92"/><line x1="100" y1="425" x2="150" y2="335"/><line x1="240" y1="438" x2="252" y2="418"/></g>
{chip('bp', 150, 22)}
{chip('sugar', 62, 135)}
{chip('chest', 360, 285)}
{chip('arteries', 75, 440)}
{chip('muscle', 245, 452)}
</svg>"""
    legend = f"<div style='font-size:12px;color:#52606d;text-align:center;margin-top:4px'>{t['legend']}</div>"
    return svg + legend

# Normal baseline reset values
NORMAL_CASE = [54, 1, 3, 120, 200, 0, 0, 150, 0, 0.0, 1, 0, 3]

def predict_factory(M):
    def predict(*vals):
        *feat_vals, lang_name = vals
        lang = "en"
        row = dict(zip(FEATURES, feat_vals))
        p, per_group = explain(M, row)
        return risk_card(p, lang), render_svg(per_group, lang), build_story(p, per_group, row, lang)
    return predict

def metrics_markdown():
    top5_text = "\n".join([f"    {idx:<3} {row['Measurement']:<10} {row['Coefficient']:.6f}" for idx, row in top5_df.reset_index().iterrows()])
    ext_text = "\n".join([res for res in ext_results])
    
    terminal_output_block = f"""```text
==========================================
1. CLEVELAND BASE MODEL & FEATURE IMPORTANCE
==========================================
Accuracy: {acc_val:.2f}
Recall: {rec_val:.2f}
Baseline Accuracy: {base_acc_val:.2f}

--- Top 5 Most Important Measurements ---
    Measurement  Coefficient
{top5_text}

==========================================
2. MULTI-HOSPITAL DOMAIN SHIFT EVALUATION
==========================================
{cle_str}
{hun_str}
{swi_str}
{lbv_str}

Selected features based on Hungary availability: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']

--- External Hospital Generalization Results ---
{ext_text}
```"""
    return f"""### Complete Model & Multi-Hospital Evaluation Output
    
{terminal_output_block}

*Clinical Prototype - For Professional Medical Evaluation Only.*"""

def build_demo(M):
    with gr.Blocks(title="CardioTriage AI - Clinical Decision Support", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🫀 CardioTriage AI: Clinical Decision Support System\n"
                    "Input patient check-up parameters to generate anatomical risk localization and professional triage recommendations.")
        with gr.Tabs():
            with gr.Tab("Patient Assessment"):
                lang = gr.Radio(["English"], value="English", label="Language", visible=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        age = gr.Slider(20, 80, value=54, step=1, label="1. age — Age in years")
                        sex = gr.Dropdown([("Male", 1), ("Female", 0)], value=1, label="2. sex — Sex")
                        cp = gr.Dropdown([("1 Typical angina", 1), ("2 Atypical angina", 2),
                                          ("3 Non-anginal pain", 3), ("4 Asymptomatic", 4)],
                                         value=3, label="3. cp — Chest pain type")
                        trestbps = gr.Slider(80, 220, value=130, step=1, label="4. trestbps — Resting blood pressure, mmHg")
                        chol = gr.Slider(100, 600, value=240, step=1, label="5. chol — Serum cholesterol, mg/dl")
                        fbs = gr.Dropdown([("No (<= 120 mg/dl)", 0), ("Yes (> 120 mg/dl)", 1)], value=0,
                                          label="6. fbs — Fasting blood sugar")
                        restecg = gr.Dropdown([("0 Normal", 0), ("1 ST-T wave abnormality", 1),
                                               ("2 Left ventricular hypertrophy", 2)], value=0,
                                              label="7. restecg — Resting ECG result")
                        thalach = gr.Slider(60, 220, value=150, step=1, label="8. thalach — Maximum heart rate achieved")
                        exang = gr.Dropdown([("No", 0), ("Yes", 1)], value=0,
                                            label="9. exang — Exercise-induced angina")
                        oldpeak = gr.Slider(0, 6.5, value=1.0, step=0.1, label="10. oldpeak — ST depression induced by exercise")
                        slope = gr.Dropdown([("1 Upsloping", 1), ("2 Flat", 2), ("3 Downsloping", 3)],
                                            value=1, label="11. slope — Peak exercise ST segment slope")
                        ca = gr.Slider(0, 3, value=0, step=1, label="12. ca — Number of major vessels colored (0-3)")
                        thal = gr.Dropdown([("3 Normal", 3), ("6 Fixed defect", 6),
                                            ("7 Reversible defect", 7)], value=3,
                                           label="13. thal — Thallium stress test result")
                        
                        btn = gr.Button("Generate Clinical Assessment", variant="primary")
                        reset_btn = gr.Button("RESET (NORMAL HUMAN VALUES)", variant="secondary")

                    with gr.Column(scale=1):
                        out_risk = gr.HTML()
                        out_svg = gr.HTML()
                        out_story = gr.HTML()

                comps = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                predict = predict_factory(M)
                outs = [out_risk, out_svg, out_story]
                
                btn.click(predict, comps + [lang], outs)
                lang.change(predict, comps + [lang], outs)
                
                # Reset button wiring
                reset_btn.click(lambda: NORMAL_CASE, None, comps).then(predict, comps + [lang], outs)
                
                demo.load(predict, comps + [lang], outs)
            with gr.Tab("Model Performance Report"):
                gr.Markdown(metrics_markdown())
    return demo

demo = build_demo(M)
demo.launch(share=True, debug=True)