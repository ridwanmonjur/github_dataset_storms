"""Per subtype, how each impact relates to hazard intensity, exposure, development and governance (Spearman and a log-log regression)
-> statistics/csv/. Run after run5: python statistics/subtype_relations.py"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).parent.parent
OUT = Path(__file__).parent / "csv"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE / "dataset_exp"))
from lib_cleans import storm_events  # noqa: E402

FIRST_YEAR, MIN_N, N_BOOT, N_BOOT_R2, SEED = 1990, 30, 1000, 300, 7
OUTCOMES = {"deaths": "deaths", "injured": "injured", "affected": "total_affected", "homeless": "homeless", "damage_usd": "total_damage_usd",
            "displaced": "displacement"}
VARIABLES = ["era5_wind_max_ms", "track_wind_kt", "area_km2", "population", "population_density", "hdi", "gdp_per_capita", "wgi_gov_effectiveness"]
REGRESSORS = ["intensity", "area_km2", "population_density", "gdp_per_capita", "year"]


def main():
    rows = storm_events()
    rows = rows[rows["year"] >= FIRST_YEAR].copy()
    assert not rows[["disasterno", "iso3"]].duplicated().any()

    # Intensity, area, country context and displacement onto event x country rows
    tracks = (pd.read_csv(BASE / "datasets" / "linked" / "track_features.csv", usecols=["disasterno", "iso3", "track_wind_kt"])
              .merge(pd.read_csv(BASE / "datasets" / "linked" / "weather_features.csv", usecols=["disasterno", "iso3", "era5_wind_max_ms"]), on=["disasterno", "iso3"], how="outer")
              .merge(pd.read_csv(BASE / "datasets" / "linked" / "exposure_features.csv", usecols=["disasterno", "iso3", "gpw_count"]), on=["disasterno", "iso3"], how="outer"))
    areas = gpd.read_file(BASE / "datasets" / "clean" / "geodisasters_areas.gpkg")
    areas["area_km2"] = areas.to_crs("+proj=cea").area / 1e6
    country = (pd.read_csv(BASE / "datasets" / "clean" / "hdi.csv", usecols=["iso3", "year", "hdi"])
               .merge(pd.read_csv(BASE / "datasets" / "clean" / "wdi.csv", usecols=["iso3", "year", "gdp_per_capita", "population"]), on=["iso3", "year"], how="outer")
               .merge(pd.read_csv(BASE / "datasets" / "clean" / "wgi.csv", usecols=["iso3", "year", "wgi_gov_effectiveness"]), on=["iso3", "year"], how="outer"))
    idmc = pd.read_csv(BASE / "datasets" / "clean" / "idmc.csv", usecols=["disasterno", "iso3", "displacement"], low_memory=False)
    disp = idmc[idmc["disasterno"].notna() & (idmc["displacement"] > 0)].groupby(["disasterno", "iso3"])["displacement"].sum().rename("displacement")
    n_rows = len(rows)
    rows = (rows.merge(tracks, on=["disasterno", "iso3"], how="left").merge(areas[["disasterno", "iso3", "area_km2"]], on=["disasterno", "iso3"], how="left")
            .merge(country, on=["iso3", "year"], how="left").merge(disp, on=["disasterno", "iso3"], how="left"))
    assert len(rows) == n_rows, "a join duplicated rows"
    # Population density of the event area: GPWv4's people over the area of the units
    rows["population_density"] = rows["gpw_count"] / rows["area_km2"]
    print(f"Event x country rows since {FIRST_YEAR}: {len(rows):,}; with track wind {rows['track_wind_kt'].notna().sum():,}, "
          f"with units {rows['area_km2'].notna().sum():,}, with GDP per capita {rows['gdp_per_capita'].notna().sum():,}")

    rng = np.random.default_rng(SEED)
    corr, regs = [], []
    for subtype, sub in rows.groupby("disaster_subtype"):
        for outcome, col in OUTCOMES.items():
            have = sub[sub[col] > 0]
            if len(have) < MIN_N:
                continue
            y = np.log(have[col])
            # 1. Spearman with each variable
            for var in VARIABLES:
                pair = pd.DataFrame({"y": y, "x": have[var]}).dropna()
                if len(pair) < MIN_N or pair["x"].nunique() < 5:
                    continue
                rho, p = spearmanr(pair["x"], pair["y"])
                idx = rng.integers(0, len(pair), size=(N_BOOT, len(pair)))
                rx, ry = rankdata(pair["x"].to_numpy()[idx], axis=1), rankdata(pair["y"].to_numpy()[idx], axis=1)
                rx, ry = rx - rx.mean(axis=1, keepdims=True), ry - ry.mean(axis=1, keepdims=True)
                with np.errstate(invalid="ignore"):
                    boot = (rx * ry).sum(axis=1) / np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum(axis=1))
                corr.append({"subtype": subtype, "outcome": outcome, "variable": var, "n": len(pair), "spearman_rho": rho,
                             "rho_lo": np.nanpercentile(boot, 2.5), "rho_hi": np.nanpercentile(boot, 97.5), "p": p})
            # 2. Log-log regression, all variables at once; intensity = IBTrACS track wind for cyclones (as Bakkensen & Mendelsohn), else EM-DAT's magnitude
            x = pd.DataFrame({"intensity": have["track_wind_kt"] if subtype == "Tropical cyclone" else have["era5_wind_max_ms"],
                              "area_km2": have["area_km2"], "population_density": have["population_density"],
                              "gdp_per_capita": have["gdp_per_capita"], "year": have["year"]})
            logs = [c for c in REGRESSORS if c in ("area_km2", "population_density", "gdp_per_capita") or (c == "intensity" and subtype == "Tropical cyclone")]
            use = [c for c in REGRESSORS if x[c].notna().sum() >= MIN_N]
            data = pd.concat([y.rename("y"), x[use]], axis=1).dropna()
            data = data[(data[[c for c in logs if c in use]] > 0).all(axis=1)]
            if len(data) < MIN_N + len(use) or not use:
                continue
            data[[c for c in logs if c in use]] = np.log(data[[c for c in logs if c in use]])
            fit = sm.OLS(data["y"], sm.add_constant(data[use])).fit(cov_type="HC3")
            ci = fit.conf_int()
            # R2 interval: the same regression refitted on resampled rows
            xm, yv = sm.add_constant(data[use]).to_numpy(), data["y"].to_numpy()
            r2s = []
            for i in rng.integers(0, len(yv), size=(N_BOOT_R2, len(yv))):
                beta = np.linalg.lstsq(xm[i], yv[i], rcond=None)[0]
                r2s.append(1 - ((yv[i] - xm[i] @ beta) ** 2).sum() / ((yv[i] - yv[i].mean()) ** 2).sum())
            for var in use:
                regs.append({"subtype": subtype, "outcome": outcome, "variable": var, "logged": var in logs, "n": int(fit.nobs),
                             "r2": fit.rsquared, "r2_adj": fit.rsquared_adj, "r2_lo": np.percentile(r2s, 2.5), "r2_hi": np.percentile(r2s, 97.5),
                             "coef": fit.params[var], "coef_lo": ci.loc[var, 0], "coef_hi": ci.loc[var, 1], "p": fit.pvalues[var]})

    corr, regs = pd.DataFrame(corr), pd.DataFrame(regs)
    for table in (corr, regs):
        table["p_holm"] = multipletests(table["p"], method="holm")[1]
        table["significant"] = table["p_holm"] < 0.05
    assert corr["spearman_rho"].between(-1, 1).all() and regs["n"].ge(MIN_N).all()
    corr.round(4).to_csv(OUT / "subtype_correlations.csv", index=False)
    regs.round(4).to_csv(OUT / "subtype_regressions.csv", index=False)

    # Which impact is most strongly connected, per subtype: the regression's adjusted R2 (with its interval), its significant coefficients,
    # and the strongest significant Spearman correlation; ranked by adjusted R2 within the subtype
    fits = regs.groupby(["subtype", "outcome"]).agg(n=("n", "first"), r2=("r2", "first"), r2_adj=("r2_adj", "first"), r2_lo=("r2_lo", "first"),
                                                    r2_hi=("r2_hi", "first"), significant_coefs=("significant", "sum"))
    sig = corr[corr["significant"]].assign(abs_rho=lambda d: d["spearman_rho"].abs())
    best = sig.loc[sig.groupby(["subtype", "outcome"])["abs_rho"].idxmax()].set_index(["subtype", "outcome"])
    fits["strongest_variable"], fits["strongest_rho"] = best["variable"], best["spearman_rho"]
    fits["significant_correlations"] = sig.groupby(["subtype", "outcome"]).size()
    fits["significant_correlations"] = fits["significant_correlations"].fillna(0).astype(int)
    fits["rank_in_subtype"] = fits.groupby("subtype")["r2_adj"].rank(ascending=False).astype(int)
    fits = fits.reset_index().sort_values(["subtype", "rank_in_subtype"])
    fits.round(4).to_csv(OUT / "subtype_outcome_connection.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(f"Spearman tests: {len(corr)}, significant after Holm: {corr['significant'].sum()}")
    print(corr[corr["significant"]].sort_values(["subtype", "outcome", "p"]).round(3).to_string(index=False))
    print(f"\nRegression coefficients: {len(regs)}, significant after Holm: {regs['significant'].sum()}")
    print(regs[regs["significant"]].sort_values(["subtype", "outcome"]).round(3).to_string(index=False))
    print("\nImpacts ranked by connection (adjusted R2) within each subtype:")
    print(fits.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
