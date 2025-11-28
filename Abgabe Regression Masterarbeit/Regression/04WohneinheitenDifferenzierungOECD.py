import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.genmod.families.links import Log

# 
FILE  = "251008_MA_16_wohneinheiten.xlsx"
XCOL  = "Bevölkerung"        
YCOL  = "Wohneinheiten Anzahl"
GCOL  = "BIP (mrd.)"
ACOL  = "Fläche qkm"         
HCOL  = "Haushalt"           
OCOL  = "OECD"               
SCOL  = "Status"

EPS   = 1e-12
SCALE = 1e6                  
N_BOOT = 1000                

# 
FEATURE_NAMES_EXT = ["const", "lnGpc", "lnA", "Hsize"]


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
def wald_print(model, R, label):
    res = model.wald_test(R, scalar=True)
    stat = float(np.atleast_1d(res.statistic)[0])
    pval = float(np.atleast_1d(res.pvalue)[0])
    df   = int(R.shape[0])
    print(f"{label}:  stat={stat:.3f}, df={df}, p={pval:.6f}")


# 
def base(df_sub):
    Xk  = df_sub[XCOL].to_numpy(float)          
    Y   = df_sub[YCOL].to_numpy(float)             
    pop = Xk
    gdp = df_sub[GCOL].to_numpy(float)             
    A   = df_sub[ACOL].to_numpy(float)             
    H   = df_sub[HCOL].to_numpy(float)             

    GDPpc = (gdp * 1e9) / np.maximum(pop, EPS)     
    lnGpc = safe_log(GDPpc)                        
    lnA   = safe_log(A)                            

    expo  = pop                                    
    return Y, expo, lnGpc, lnA, H, Xk


# 
def make_design(df_sub):
    """
    Erzeugt Zielvariable, Exposure und Designmatrix X für NB2.
    Regressoren: lnGpc (log BIP je Kopf), lnA (log Fläche), Haushaltsgröße (Hsize).
    """
    y, expo, lnGpc, lnA, H, Xk = base(df_sub)
    X = sm.add_constant(np.column_stack([lnGpc, lnA, H]))
    return y, expo, X, Xk


# 
def bootstrap_df(df_sub, n_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(df_sub), size=n_samples)
    return df_sub.iloc[idx].reset_index(drop=True)


# 
def run_group(df_tr_group, df_val_group, label):
    print("\n" + "="*80)
    print(f"=== MODELL FÜR GRUPPE: {label} ===")

    if len(df_tr_group) == 0:
        print("Keine Trainingsdaten für diese Gruppe – Modell kann nicht geschätzt werden.")
        return

    print(f"Train-Samples (original, {label}): {len(df_tr_group)}")
    print(f"Validierungs-Samples ({label}):     {len(df_val_group)}")

    # 
    df_tr_boot = bootstrap_df(df_tr_group, n_samples=N_BOOT)
    print(f"Train-Samples (bootstrap, {label}): {len(df_tr_boot)}")

    # 
    y_tr, expo_tr, X_tr, Xk_tr = make_design(df_tr_boot)
    y_tr_s = y_tr / SCALE

    nb_boot, alpha_hat_boot = fit_nb2_with_alpha(X_tr, y_tr_s, expo_tr)

    print(f"\n--- NB2 (Bootstrap, {label}) — Formel (Offset log(Bev.), Regressoren: lnGpc, lnA(Fläche), Haushaltsgröße) ---")
    print(f"alpha (Dispersion) = {alpha_hat_boot:.6g}")
    print(coef_table(nb_boot, FEATURE_NAMES_EXT).to_string(float_format=lambda x: f"{x:,.6g}"))

    # 
    r2_mcf, ll, ll0, aic = mcfadden_r2(nb_boot, X_tr, y_tr_s, expo_tr, alpha_hat_boot)
    print(f"LogLik={ll:,.3f} | LogLik(null)={ll0:,.3f} | AIC={aic:,.3f} | McFadden R^2={r2_mcf:.4f}")

    # 
    yhat_tr = nb_boot.fittedvalues * SCALE

    print(f"\n=== [NB2 (Bootstrap, {label}) — Metriken (Train)] ===")
    print_reg_metrics(f"Train NB2 Boot ({label})", reg_metrics(y_tr, yhat_tr))

    # 
    print(f"\n=== [Wald-Tests auf Signifikanz (NB2 Bootstrap, {label})] ===")
    name_to_idx_ext = {n: i for i, n in enumerate(FEATURE_NAMES_EXT)}

    R_lngpc = np.zeros((1, len(FEATURE_NAMES_EXT))); R_lngpc[0, name_to_idx_ext["lnGpc"]]  = 1.0
    R_lnA   = np.zeros((1, len(FEATURE_NAMES_EXT))); R_lnA[0,   name_to_idx_ext["lnA"]]    = 1.0
    R_H     = np.zeros((1, len(FEATURE_NAMES_EXT))); R_H[0,     name_to_idx_ext["Hsize"]]  = 1.0

    R_joint = np.zeros((3, len(FEATURE_NAMES_EXT)))
    R_joint[0, name_to_idx_ext["lnGpc"]] = 1.0
    R_joint[1, name_to_idx_ext["lnA"]]   = 1.0
    R_joint[2, name_to_idx_ext["Hsize"]] = 1.0

    wald_print(nb_boot, R_joint, "Gemeinsam lnGpc=lnA=Hsize=0")
    wald_print(nb_boot, R_lngpc, "Einzeln lnGpc=0 (log BIP je Kopf)")
    wald_print(nb_boot, R_lnA,   "Einzeln lnA=0 (log Fläche)")
    wald_print(nb_boot, R_H,     "Einzeln Hsize=0 (Haushaltsgröße)")

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
    plt.title(f"NB2-GLM (Bootstrap, {label}) — Observed vs Predicted (Training)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    # 
    if len(df_val_group) > 0:
        y_val, expo_val, X_val, Xk_val = make_design(df_val_group)

        yhat_val_s = nb_boot.predict(X_val, exposure=expo_val)
        yhat_val   = yhat_val_s * SCALE

        print(f"\n=== [NB2 (Bootstrap, {label}) — Metriken (Validation)] ===")
        print_reg_metrics(f"Val NB2 Boot ({label})", reg_metrics(y_val, yhat_val))

        plt.figure(figsize=(10, 5))
        plt.scatter(Xk_val, y_val,    s=18, label="Observed (Validation)", color="tab:blue")
        plt.scatter(
            Xk_val, yhat_val, s=18,
            facecolors="none", edgecolors="tab:blue",
            label="NB2 ŷ (Validation)"
        )
        plt.xlabel(XCOL)
        plt.ylabel(YCOL)
        plt.title(f"NB2-GLM (Bootstrap, {label}) — Observed vs Predicted (Validation)")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()
    else:
        print(f"\nHinweis: Es wurden keine Validierungsdaten für {label} gefunden (Status ≠ 'training').")


# 
df = pd.read_excel(FILE, engine="openpyxl")

status_norm = df[SCOL].astype(str).str.normalize("NFKC").str.strip().str.casefold()
is_train = status_norm == "training"
is_val   = ~is_train

# 
cols_used = [XCOL, YCOL, GCOL, ACOL, HCOL, OCOL]

df_tr_all  = df.loc[is_train, cols_used].copy()
df_val_all = df.loc[is_val,   cols_used].copy()

for c in cols_used:
    df_tr_all[c]  = pd.to_numeric(df_tr_all[c],  errors="coerce")
    df_val_all[c] = pd.to_numeric(df_val_all[c], errors="coerce")

df_tr_all  = df_tr_all.dropna()
df_val_all = df_val_all.dropna()

# 
df_tr_oecd     = df_tr_all[df_tr_all[OCOL] == 1].copy()
df_tr_non_oecd = df_tr_all[df_tr_all[OCOL] == 0].copy()

df_val_oecd     = df_val_all[df_val_all[OCOL] == 1].copy()
df_val_non_oecd = df_val_all[df_val_all[OCOL] == 0].copy()

# 
for d in (df_tr_oecd, df_tr_non_oecd, df_val_oecd, df_val_non_oecd):
    if OCOL in d.columns:
        d.drop(columns=[OCOL], inplace=True)

# 
run_group(df_tr_oecd,     df_val_oecd,     label="OECD (OECD=1)")
run_group(df_tr_non_oecd, df_val_non_oecd, label="Nicht-OECD (OECD=0)")
