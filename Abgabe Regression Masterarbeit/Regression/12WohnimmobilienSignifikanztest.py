import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.families.links import Log

# 
FILE  = "251008_MA_17_Wohnimmobilien.xlsx"
XCOL  = "Bevölkerung"        
YCOL  = "Wohnimmobilien Anzahl"
GCOL  = "BIP (mrd.)"
UCOL  = "Urbanisierung"      
ECOL  = "Strom (mrd.)"       
ACOL  = "Fläche qkm"         
HCOL  = "Haushalt"           
SCOL  = "Status"

EPS   = 1e-12
SCALE = 1e6                  
N_BOOT = 500                 

#
def safe_log(x):
    return np.log(np.maximum(x, EPS))


def coef_table(model, names=None):
    beta = np.asarray(model.params); se = np.asarray(model.bse); k = beta.shape[0]
    if names is None or len(names) != k:
        names = [f"b{i}" for i in range(k)]
    zval = getattr(model, "tvalues", beta / np.where(se == 0, np.nan, se))
    pval = getattr(model, "pvalues", np.full(k, np.nan))
    ci = np.asarray(model.conf_int())
    if ci.ndim == 1:
        ci = ci.reshape(-1, 2)
    if ci.shape[0] != k:
        tmp = np.full((k, 2), np.nan)
        tmp[:ci.shape[0], :ci.shape[1]] = ci
        ci = tmp
    return pd.DataFrame(
        {
            "beta": beta,
            "std.err": se,
            "z": np.asarray(zval),
            "p>|z|": np.asarray(pval),
            "CI low": ci[:, 0],
            "CI high": ci[:, 1],
        },
        index=names,
    )


def mcfadden_r2(model, X, y_s, expo, alpha):
    X0 = sm.add_constant(np.zeros((X.shape[0], 0)))
    fam = sm.families.NegativeBinomial(alpha=alpha, link=Log())
    null = sm.GLM(y_s, X0, family=fam, exposure=expo).fit(maxiter=300)
    ll0 = float(null.llf)
    ll  = float(model.llf)
    aic = float(model.aic)
    r2  = 1.0 - ll / ll0 if ll0 != 0 else np.nan
    return r2, ll, ll0, aic


def fit_nb2_with_alpha(X, y_s, expo):
    pois0 = sm.GLM(y_s, X, family=sm.families.Poisson(link=Log()),
                   exposure=expo).fit(maxiter=300)
    best = None
    for a in np.logspace(-3, 1, 40):
        try:
            nb_try = sm.GLM(
                y_s, X,
                family=sm.families.NegativeBinomial(alpha=a, link=Log()),
                exposure=expo
            ).fit(start_params=pois0.params, maxiter=700)
            if best is None or nb_try.llf > best[0]:
                best = (nb_try.llf, a, nb_try.params)
        except Exception:
            pass
    if best is None:
        raise RuntimeError("NB2 alpha search failed.")
    a_hat = float(best[1])
    nb = sm.GLM(
        y_s, X,
        family=sm.families.NegativeBinomial(alpha=a_hat, link=Log()),
        exposure=expo
    ).fit(cov_type="HC3", start_params=best[2], maxiter=900)
    return nb, a_hat


# 
def reg_metrics(y, yhat):
    r    = y - yhat
    mse  = float(np.mean(r**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(r)))
    mape = float(np.mean(np.abs(r) / np.maximum(np.abs(y), EPS)) * 100)
    bias = float(np.mean(r))

    ss_res = float(np.sum(r**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return mse, rmse, mae, mape, bias, r2


def print_reg_metrics(tag, m):
    print(
        f"{tag} | "
        f"MSE={m[0]:,.2f} | RMSE={m[1]:,.2f} | "
        f"MAE={m[2]:,.2f} | MAPE={m[3]:.2f}% | "
        f"Bias={m[4]:,.2f} | R²={m[5]:.4f}"
    )


# 
df = pd.read_excel(FILE, engine="openpyxl")

status_norm = df[SCOL].astype(str).str.normalize("NFKC").str.strip().str.casefold()
is_train = status_norm == "training"
is_val   = ~is_train

cols_used = [XCOL, YCOL, GCOL, UCOL, ECOL, ACOL, HCOL]

df_tr  = df.loc[is_train, cols_used].copy()
df_val = df.loc[is_val,   cols_used].copy()

for c in cols_used:
    df_tr[c]  = pd.to_numeric(df_tr[c],  errors="coerce")
    df_val[c] = pd.to_numeric(df_val[c], errors="coerce")

df_tr  = df_tr.dropna()
df_val = df_val.dropna()

print(f"Train-Samples (original): {len(df_tr)}")
print(f"Validierungs-Samples:     {len(df_val)}")


# 
def base(df_sub):
    Xk  = df_sub[XCOL].to_numpy(float)             
    Y   = df_sub[YCOL].to_numpy(float)             
    pop = Xk
    gdp = df_sub[GCOL].to_numpy(float)             
    U   = np.clip(df_sub[UCOL].to_numpy(float), 0, 1)
    E   = df_sub[ECOL].to_numpy(float)             
    A   = df_sub[ACOL].to_numpy(float)             
    H   = df_sub[HCOL].to_numpy(float)             

    GDPpc = (gdp * 1e9) / np.maximum(pop, EPS)     
    BIPpk = GDPpc                                  
    lnE   = safe_log(E)
    lnA   = safe_log(A)

    expo  = pop
    return Y, expo, BIPpk, U, lnE, lnA, H, Xk


# 
def make_design(df_sub, features):
    """
    features: Teilmenge aus ["BIPpk", "U", "lnE", "lnA", "Hsize"]
    """
    y, expo, BIPpk, U, lnE, lnA, H, Xk = base(df_sub)
    data = {
        "BIPpk": BIPpk,
        "U": U,
        "lnE": lnE,
        "lnA": lnA,
        "Hsize": H,
    }

    cols = []
    for f in features:
        cols.append(data[f])

    X = sm.add_constant(np.column_stack(cols))
    feat_names = ["const"] + list(features)
    return y, expo, X, Xk, feat_names


# 
def bootstrap_df(df_sub, n_samples=500, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(df_sub), size=n_samples)
    return df_sub.iloc[idx].reset_index(drop=True)


df_tr_boot = bootstrap_df(df_tr, n_samples=N_BOOT)
print(f"Train-Samples (bootstrap): {len(df_tr_boot)}")


# 
feature_sets = {
    "Vollmodell":            ["BIPpk", "U", "lnE", "lnA", "Hsize"],
    "ohne BIPpk":            ["U", "lnE", "lnA", "Hsize"],
    "ohne U":                ["BIPpk", "lnE", "lnA", "Hsize"],
    "ohne lnE":              ["BIPpk", "U", "lnA", "Hsize"],
    "ohne lnA":              ["BIPpk", "U", "lnE", "Hsize"],
    "ohne Hsize":            ["BIPpk", "U", "lnE", "lnA"],
}

results = {}

for label, feats in feature_sets.items():
    print("\n" + "="*90)
    print(f"=== Modell: {label} — Regressoren: {', '.join(feats)} ===")

    
    y_tr, expo_tr, X_tr, Xk_tr, feat_names = make_design(df_tr_boot, feats)
    y_tr_s = y_tr / SCALE

    
    nb, alpha_hat = fit_nb2_with_alpha(X_tr, y_tr_s, expo_tr)

    print(f"alpha (Dispersion) = {alpha_hat:.6g}")
    print(coef_table(nb, feat_names).to_string(float_format=lambda x: f"{x:,.6g}"))

    
    r2_mcf, ll, ll0, aic = mcfadden_r2(nb, X_tr, y_tr_s, expo_tr, alpha_hat)
    print(f"LogLik={ll:,.3f} | LogLik(null)={ll0:,.3f} | "
          f"AIC={aic:,.3f} | McFadden R^2={r2_mcf:.4f}")

    
    yhat_tr = nb.fittedvalues * SCALE
    metrics_tr = reg_metrics(y_tr, yhat_tr)
    print("\n[Trainingsmetriken]")
    print_reg_metrics("Train NB2", metrics_tr)

    
    metrics_val = None
    if len(df_val) > 0:
        y_val, expo_val, X_val, Xk_val, _ = make_design(df_val, feats)
        yhat_val_s = nb.predict(X_val, exposure=expo_val)
        yhat_val   = yhat_val_s * SCALE
        metrics_val = reg_metrics(y_val, yhat_val)
        print("\n[Validierungsmetriken]")
        print_reg_metrics("Val NB2", metrics_val)
    else:
        print("\nHinweis: Keine Validierungsdaten verfügbar.")

    results[label] = {
        "features": feats,
        "model": nb,
        "alpha": alpha_hat,
        "feat_names": feat_names,
        "y_tr": y_tr,
        "yhat_tr": yhat_tr,
        "Xk_tr": Xk_tr,
        "metrics_tr": metrics_tr,
        "metrics_val": metrics_val,
    }


voll_label = "Vollmodell"
if voll_label in results:
    res = results[voll_label]
    Xk_tr_f = res["Xk_tr"]
    y_tr_f  = res["y_tr"]
    yhat_tr_f = res["yhat_tr"]

    plt.figure(figsize=(10, 5))
    plt.scatter(Xk_tr_f, y_tr_f, s=18,
                label="Observed (Train, Boot)", color="tab:blue")
    plt.scatter(
        Xk_tr_f, yhat_tr_f, s=18,
        facecolors="none", edgecolors="tab:blue",
        label="NB2 ŷ (Train, Boot)"
    )
    plt.xlabel(XCOL)
    plt.ylabel(YCOL)
    plt.title("NB2-GLM (Bootstrap, Vollmodell) — Observed vs Predicted (Training)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
