import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.families.links import Log

# 
FILE  = "251008_MA_18_NichtWohnen.xlsx"  
XCOL  = "Bevölkerung"                    
YCOL  = "Nicht Wohngebäude Anzahl"       
ECOL  = "Strom (mrd.)"                   
SCOL  = "Status"                         

EPS   = 1e-12
SCALE = 1e6                              
N_BOOT = 1000                            

FEATURE_NAMES_EXT = ["const", "E"]       


# 
def safe_log(x):
    return np.log(np.maximum(x, EPS))


def coef_table(model, names=None):
    beta = np.asarray(model.params)
    se   = np.asarray(model.bse)
    k    = beta.shape[0]

    if names is None or len(names) != k:
        names = [f"b{i}" for i in range(k)]

    zval = getattr(model, "tvalues", beta / np.where(se == 0, np.nan, se))
    pval = getattr(model, "pvalues", np.full(k, np.nan))
    ci   = np.asarray(model.conf_int())

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
    """Poisson-Start, dann alpha-Suche für NB2."""
    pois0 = sm.GLM(
        y_s, X,
        family=sm.families.Poisson(link=Log()),
        exposure=expo
    ).fit(maxiter=300)

    best = None
    for a in np.logspace(-3, 1, 40):
        try:
            nb_try = sm.GLM(
                y_s,
                X,
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
        y_s,
        X,
        family=sm.families.NegativeBinomial(alpha=a_hat, link=Log()),
        exposure=expo
    ).fit(cov_type="HC3", start_params=best[2], maxiter=900)

    return nb, a_hat


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


def wald_print(model, R, label):
    res = model.wald_test(R, scalar=True)
    stat = float(np.atleast_1d(res.statistic)[0])
    pval = float(np.atleast_1d(res.pvalue)[0])
    df   = int(R.shape[0])
    print(f"{label}:  stat={stat:.3f}, df={df}, p={pval:.6f}")


# 
df = pd.read_excel(FILE, engine="openpyxl")

status_norm = df[SCOL].astype(str).str.normalize("NFKC").str.strip().str.casefold()
is_train = status_norm == "training"
is_val   = ~is_train

cols_used = [XCOL, YCOL, ECOL]

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
    """
    - Y: Zielvariable (Nichtwohnimmobilien)
    - expo: Offset (Bevölkerung)
    - E: Strom (Niveau)
    - Xk: Bevölkerung (für Plots)
    """
    Xk  = df_sub[XCOL].to_numpy(float)
    Y   = df_sub[YCOL].to_numpy(float)
    pop = Xk
    E   = df_sub[ECOL].to_numpy(float)

    expo = pop
    return Y, expo, E, Xk


def make_design(df_sub):
    y, expo, E, Xk = base(df_sub)
    X = sm.add_constant(E.reshape(-1, 1), has_constant="add")
    return y, expo, X, Xk


# 
def bootstrap_df(df_sub, n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(df_sub), size=n_samples)
    return df_sub.iloc[idx].reset_index(drop=True)


df_tr_boot = bootstrap_df(df_tr, n_samples=N_BOOT)
print(f"Train-Samples (bootstrap): {len(df_tr_boot)}")


# 
y_tr, expo_tr, X_tr, Xk_tr = make_design(df_tr_boot)
y_tr_s = y_tr / SCALE

nb_boot, alpha_hat_boot = fit_nb2_with_alpha(X_tr, y_tr_s, expo_tr)

print("\n--- NB2 (Bootstrap) — Formel (Offset log(Bev.), Regressor: Strom E) ---")
print(f"alpha (Dispersion) = {alpha_hat_boot:.6g}")
print(coef_table(nb_boot, FEATURE_NAMES_EXT).to_string(float_format=lambda x: f"{x:,.6g}"))

r2_mcf, ll, ll0, aic = mcfadden_r2(nb_boot, X_tr, y_tr_s, expo_tr, alpha_hat_boot)
print(f"LogLik={ll:,.3f} | LogLik(null)={ll0:,.3f} | AIC={aic:,.3f} | McFadden R^2={r2_mcf:.4f}")

# 
yhat_tr = nb_boot.fittedvalues * SCALE
print("\n=== [NB2 (Bootstrap) — Metriken (Train)] ===")
print_reg_metrics("Train NB2 (E)", reg_metrics(y_tr, yhat_tr))


# 
print("\n=== [Wald-Test auf Signifikanz (E)] ===")
R_E = np.zeros((1, len(FEATURE_NAMES_EXT)))
R_E[0, FEATURE_NAMES_EXT.index("E")] = 1.0
wald_print(nb_boot, R_E, "Einzeln E=0 (Strom)")


# 
plt.figure(figsize=(10, 5))
plt.scatter(Xk_tr, y_tr,    s=18, label="Observed (Train, Boot)", color="tab:blue")
plt.scatter(
    Xk_tr, yhat_tr, s=18,
    facecolors="none", edgecolors="tab:blue",
    label="NB2 ŷ (Train, Boot)"
)
plt.xlabel(XCOL)
plt.ylabel(YCOL)
plt.title("NB2-GLM Nichtwohn (Bootstrap) — Observed vs Predicted (Training, nur Strom)")
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()


# 
if len(df_val) > 0:
    y_val, expo_val, X_val, Xk_val = make_design(df_val)

    yhat_val_s = nb_boot.predict(X_val, exposure=expo_val)
    yhat_val   = yhat_val_s * SCALE

    print("\n=== [NB2 (Bootstrap) — Metriken (Validation)] ===")
    print_reg_metrics("Val NB2 (E)", reg_metrics(y_val, yhat_val))

    plt.figure(figsize=(10, 5))
    plt.scatter(Xk_val, y_val,    s=18, label="Observed (Validation)", color="tab:blue")
    plt.scatter(
        Xk_val, yhat_val, s=18,
        facecolors="none", edgecolors="tab:blue",
        label="NB2 ŷ (Validation)"
    )
    plt.xlabel(XCOL)
    plt.ylabel(YCOL)
    plt.title("NB2-GLM Nichtwohn (Bootstrap) — Observed vs Predicted (Validation, nur Strom)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
else:
    print("\nHinweis: Es wurden keine Validierungsdaten gefunden (Status ≠ 'training').")
