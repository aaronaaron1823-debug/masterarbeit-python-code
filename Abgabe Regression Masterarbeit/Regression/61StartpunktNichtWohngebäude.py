import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.families.links import Log

# 
FILE  = "251008_MA_18_NichtWohnen.xlsx"
XCOL  = "Bevölkerung"        
YCOL  = "Nicht Wohngebäude Anzahl"
GCOL  = "BIP (mrd.)"
UCOL  = "Urbanisierung"      
ECOL  = "Strom (mrd.)"       
ACOL  = "Fläche qkm"
HCOL  = "Haushalt"
SCOL  = "Status"

EPS   = 1e-12
SCALE = 1e6
N_BOOT = 1000

# 
REGRESSORS = ["BIPpk", "U", "E", "lnA", "Hsize"]


# 
def safe_log(x):
    return np.log(np.maximum(x, EPS))


def coef_table(model, names=None):
    beta = np.asarray(model.params); se = np.asarray(model.bse); k = beta.shape[0]
    if names is None or len(names) != k:
        names = [f"b{i}" for i in range(k)]
    zval = getattr(model, "tvalues", beta / np.where(se==0, np.nan, se))
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
    pois0 = sm.GLM(y_s, X, family=sm.families.Poisson(link=Log()), exposure=expo).fit(maxiter=300)
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
    lnA   = safe_log(A)

    expo  = pop

    reg_dict = {
        "BIPpk": BIPpk,
        "U":     U,
        "E":     E,
        "lnA":   lnA,
        "Hsize": H,
    }

    return Y, expo, reg_dict, Xk


def make_design_single(df_sub, reg_name):
    y, expo, reg_dict, Xk = base(df_sub)
    r = np.asarray(reg_dict[reg_name], float).reshape(-1, 1)
  
    X = sm.add_constant(r, has_constant="add")
    return y, expo, X, Xk


# 
def bootstrap_df(df_sub, n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(df_sub), size=n_samples)
    return df_sub.iloc[idx].reset_index(drop=True)


df_tr_boot = bootstrap_df(df_tr, n_samples=N_BOOT)
print(f"Train-Samples (bootstrap): {len(df_tr_boot)}")


# 
results = []

for reg in REGRESSORS:
    print("\n" + "="*80)
    print(f"=== NB2 (Bootstrap) — Ein-Regressor-Modell: {reg} ===")

  
    y_tr, expo_tr, X_tr, Xk_tr = make_design_single(df_tr_boot, reg)
    y_tr_s = y_tr / SCALE

    nb_mod, alpha_hat = fit_nb2_with_alpha(X_tr, y_tr_s, expo_tr)

    feat_names = ["const", reg]
    print(f"\nalpha (Dispersion) = {alpha_hat:.6g}")
    print(coef_table(nb_mod, feat_names).to_string(float_format=lambda x: f"{x:,.6g}"))

    r2_mcf, ll, ll0, aic = mcfadden_r2(nb_mod, X_tr, y_tr_s, expo_tr, alpha_hat)
    print(f"\nLogLik={ll:,.3f} | LogLik(null)={ll0:,.3f} | AIC={aic:,.3f} | McFadden R^2={r2_mcf:.4f}")

    yhat_tr = nb_mod.fittedvalues * SCALE
    metr_tr = reg_metrics(y_tr, yhat_tr)
    print("\n--- Metriken (Train) ---")
    print_reg_metrics(f"Train NB2 ({reg})", metr_tr)

    print("\n--- Wald-Test ---")
    R = np.zeros((1, len(feat_names))); R[0, 1] = 1.0
    wald_print(nb_mod, R, f"Einzeln {reg}=0")

  
    if len(df_val) > 0:
        y_val, expo_val, X_val, Xk_val = make_design_single(df_val, reg)
        yhat_val_s = nb_mod.predict(X_val, exposure=expo_val)
        yhat_val   = yhat_val_s * SCALE
        metr_val   = reg_metrics(y_val, yhat_val)

        print("\n--- Metriken (Validation) ---")
        print_reg_metrics(f"Val NB2 ({reg})", metr_val)

        val_mape = metr_val[3]
        val_r2   = metr_val[5]
    else:
        val_mape = np.nan
        val_r2   = np.nan
        print("\nHinweis: Keine Validierungsdaten vorhanden (Status ≠ 'training').")

    results.append({
        "Regressor":    reg,
        "alpha":        alpha_hat,
        "AIC":          aic,
        "McFadden_R2":  r2_mcf,
        "Train_MAPE":   metr_tr[3],
        "Train_R2":     metr_tr[5],
        "Val_MAPE":     val_mape,
        "Val_R2":       val_r2,
    })

# 
print("\n" + "="*80)
print("=== Vergleich der Ein-Regressor-Modelle ===")
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
