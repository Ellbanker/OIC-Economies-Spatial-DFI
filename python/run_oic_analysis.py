#ahmed alakrawi
#!/usr/bin/env python3
"""Reproducible analysis for the ORIGINAL OIC spatial--Mundlak manuscript.

Original manuscript: 'Digital Financial Inclusion, Governance, and GDP per
Capita in OIC Economies: A Spatial--Mundlak Panel Analysis', 37 countries,
2017--2023 (259 country-years), Eq. (6) in the provided source.

Modes:
  baseline  : estimable non-spatial Mundlak/year-FE and two-way country/year-FE
              checks from supplied DFI/Gov/GDPpc data. NOT a spatial estimate.
  spatial   : spatial autoregressive (SAR) maximum-likelihood restricted Durbin
              model with Mundlak terms and year indicators. REQUIRES the paper's
              weights matrix or its trade openness/income-class inputs.

No missing source indicators, index-construction steps or regressions are
invented. Year indicators and SML are reviewer-motivated modifications. The
2017--2023 country averaging window is an explicit reproducible assumption;
the original methods' D12 placeholder must still be confirmed by the authors.

Dependencies: numpy, scipy, statsmodels. CSV inputs need no Excel packages.
To read .xlsx directly, additionally install/use artifact_tool; a ready-to-use
CSV conversion of the uploaded workbook is bundled in the project.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import optimize, stats
import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_hess

YEARS = tuple(range(2017, 2024))
EXCLUDED = {"Afghanistan", "Libya", "Sudan"}
EPSILON = 1e-5  # The paper's inverse-distance stabilizing constant
REQUIRED = ("Country", "Year", "DFI_Index", "Gov", "GDPpc")


def load_rows(path: Path) -> list[dict]:
    """Read the original Excel workbook using artifact_tool or standard CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() in (".csv", ".txt"):
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() == ".xlsx":
        try:
            from artifact_tool import Blob, SpreadsheetFile
        except ImportError as exc:
            raise RuntimeError("For XLSX input use artifact_tool, or use the bundled CSV file instead") from exc
        book = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
        metadata = book.inspect({"kind": "sheet", "include": "id,name"}).ndjson
        sheet_info = json.loads(metadata.splitlines()[0])
        sheet = book.worksheets.get_item_at(0)
        matrix = sheet.get_range(sheet_info["range"]).values
        headers = [str(v).strip() for v in matrix[0]]
        return [dict(zip(headers, row)) for row in matrix[1:] if any(v is not None for v in row)]
    raise ValueError("Unsupported file type: use .csv or .xlsx")


def write_csv(path: Path, rows: list[dict], fields=None):
    if not rows:
        raise ValueError(f"No data to save: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def number(record: dict, key: str, context: str) -> float:
    raw = record.get(key)
    if raw is None or str(raw).strip() == "":
        raise ValueError(f"Missing {key} at {context}")
    try:
        x = float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid numeric {key}={raw!r} at {context}") from e
    if not np.isfinite(x):
        raise ValueError(f"Non-finite {key} at {context}")
    return x


def load_panel(data_path: Path):
    raw = load_rows(data_path)
    if not raw:
        raise ValueError("No observations in input")
    missing_cols = set(REQUIRED) - set(raw[0])
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    full_country_count = len(set(str(r["Country"]).strip() for r in raw))
    panel = {}
    for r in raw:
        country = str(r["Country"]).strip()
        year = int(number(r, "Year", country))
        if country in EXCLUDED or year not in YEARS:
            continue
        key = (country, year)
        if key in panel:
            raise ValueError(f"Duplicate country-year: {key}")
        record = {
            "Country": country,
            "Year": year,
            "DFI_Index": number(r, "DFI_Index", str(key)),
            "Gov": number(r, "Gov", str(key)),
            "GDPpc": number(r, "GDPpc", str(key)),
        }
        if record["GDPpc"] <= 0:
            raise ValueError(f"GDPpc must be >0: {key}")
        record["ln_GDPpc"] = math.log(record["GDPpc"])
        panel[key] = record

    countries = sorted({name for (name, _year) in panel})
    if len(countries) != 37 or len(panel) != 259:
        raise ValueError(
            f"Cannot reproduce paper sample: got {len(countries)} countries, "
            f"{len(panel)} observations; expected 37 and 259. "
            "Paper excludes Afghanistan, Libya, Sudan and uses 2017--2023."
        )
    for country in countries:
        absent = set(YEARS) - {yr for c, yr in panel if c == country}
        if absent:
            raise ValueError(f"Unbalanced panel for {country}: missing {sorted(absent)}")

    # Year-major ordering is essential for repeated W network calculations.
    records = [panel[(country, year)] for year in YEARS for country in countries]
    return records, countries, full_country_count


def enrich_with_supplement(records, path: Path | None):
    """Optional country-year Trade_Op_pct + baseline Income_Level data.

    Original sheet contains neither. These fields MUST be supplied, not guessed
    from ED, IF, Edu, Gov, GDPpc or contemporaneous income thresholds.
    """
    if path is None:
        return False, False
    rows = load_rows(path)
    if not rows:
        raise ValueError("Supplement is empty")
    if not {"Country", "Year"}.issubset(rows[0]):
        raise ValueError("Supplement must contain Country,Year")
    have_trade = "Trade_Op_pct" in rows[0]
    have_income = "Income_Level" in rows[0]
    if not (have_trade or have_income):
        raise ValueError("Supplement must have Trade_Op_pct and/or Income_Level columns")
    mapping = {}
    for r in rows:
        country = str(r["Country"]).strip()
        year = int(number(r, "Year", country))
        key = (country, year)
        if key in mapping:
            raise ValueError(f"Duplicate supplement country-year: {key}")
        mapping[key] = r
    for rec in records:
        key = (rec["Country"], rec["Year"])
        if key not in mapping:
            raise ValueError(f"Missing supplement country-year {key}")
        supp = mapping[key]
        if have_trade:
            rec["Trade_Op_pct"] = number(supp, "Trade_Op_pct", str(key))
        if have_income:
            val = number(supp, "Income_Level", str(key))
            if val not in (0, 1):
                raise ValueError(f"Income_Level must be binary 0/1 at {key}")
            rec["Income_Level"] = int(val)
    if have_income:
        by_country = {}
        for rec in records:
            by_country.setdefault(rec["Country"], set()).add(rec["Income_Level"])
        varying = [c for c, values in by_country.items() if len(values) != 1]
        if varying:
            raise ValueError(f"Income_Level must be fixed over years: {varying}")
    return have_trade, have_income


def descriptives(records):
    summary = {}
    for col in ("DFI_Index", "Gov", "GDPpc", "ln_GDPpc"):
        x = np.array([r[col] for r in records])
        summary[col] = {
            "n": len(x), "mean": float(x.mean()), "sd_sample": float(x.std(ddof=1)),
            "min": float(x.min()), "max": float(x.max()),
        }
    return summary


def make_mundlak(records, countries, include_income=False):
    means = {}
    for country in countries:
        subset = [r for r in records if r["Country"] == country]
        means[country] = {
            "DFI": float(np.mean([r["DFI_Index"] for r in subset])),
            "Gov": float(np.mean([r["Gov"] for r in subset])),
        }
    for r in records:
        r["DFI_between"] = means[r["Country"]]["DFI"]
        r["DFI_within"] = r["DFI_Index"] - r["DFI_between"]
        r["Gov_between"] = means[r["Country"]]["Gov"]
        r["Gov_within"] = r["Gov"] - r["Gov_between"]
    names = ["const", "DFI_within", "DFI_between", "Gov_within", "Gov_between"]
    if include_income:
        names.append("Income_Level")
    names += [f"Year_{y}" for y in YEARS[1:]]
    X = np.array([
        [1, r["DFI_within"], r["DFI_between"], r["Gov_within"], r["Gov_between"]]
        + ([r["Income_Level"]] if include_income else [])
        + [float(r["Year"] == year) for year in YEARS[1:]]
        for r in records
    ], dtype=float)
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Mundlak model matrix is rank-deficient")
    return X, names


def regression_table(results, names):
    se = np.asarray(results.bse)
    betas = np.asarray(results.params)
    pvals = np.asarray(results.pvalues)
    conf = np.asarray(results.conf_int(alpha=.05))
    return [
        {"term": name, "coefficient": float(betas[j]), "std_error": float(se[j]),
         "p_value": float(pvals[j]), "ci_low": float(conf[j, 0]),
         "ci_high": float(conf[j, 1])}
        for j, name in enumerate(names)
    ]


def ols_checks(records, countries, include_income, out_dir):
    y = np.array([r["ln_GDPpc"] for r in records], float)
    groups = np.array([countries.index(r["Country"]) for r in records], int)
    X, names = make_mundlak(records, countries, include_income)
    ols = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}, use_t=True
    )
    mundlak_rows = regression_table(ols, names)
    write_csv(out_dir / "nonspatial_mundlak_yearFE.csv", mundlak_rows)

    # Two-way FE checks the within effects while absorbing all time-invariant
    # country characteristics, and common year shocks. NOT a spatial model.
    baseline_country = countries[0]
    fe_names = ["const", "DFI_Index", "Gov"]
    fe_names += [f"Year_{y}" for y in YEARS[1:]]
    fe_names += [f"Country_{c}" for c in countries[1:]]
    Xfe = np.array([
        [1, r["DFI_Index"], r["Gov"]]
        + [float(r["Year"] == year) for year in YEARS[1:]]
        + [float(r["Country"] == country) for country in countries[1:]]
        for r in records
    ], float)
    assert np.linalg.matrix_rank(Xfe) == Xfe.shape[1]
    fe = sm.OLS(y, Xfe).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}, use_t=True
    )
    twfe_rows = regression_table(fe, fe_names)
    write_csv(out_dir / "two_way_FE_yearFE.csv", twfe_rows)
    return {
        "nonspatial_mundlak_yearFE": {"n": len(y), "r_squared": float(ols.rsquared),
                                      "cluster_count": len(countries), "coefficients": mundlak_rows},
        "two_way_FE_yearFE": {"n": len(y), "r_squared": float(fe.rsquared),
                               "cluster_count": len(countries), "base_country": baseline_country,
                               "coefficients": twfe_rows},
    }


def build_W_from_trade(records, countries, standardize=False):
    """Equation (3)-(5): inverse Euclidean distance +1e-5, row standardized.

    Unstandardized coordinates are the ORIGINAL manuscript's definition. The
    standardized option is a SEPARATE model for sensitivity, not a replication.
    """
    avg_trade = np.array([
        np.mean([r["Trade_Op_pct"] for r in records if r["Country"] == country])
        for country in countries
    ], float)
    income = np.array([
        next(r["Income_Level"] for r in records if r["Country"] == country)
        for country in countries
    ], float)
    coords = np.column_stack((avg_trade, income))
    if standardize:
        std = coords.std(axis=0, ddof=1)
        if np.any(std <= 0):
            raise ValueError("Standardized weight sensitivity requires both coordinates vary")
        coords = (coords - coords.mean(axis=0)) / std
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    A = 1.0 / (distances + EPSILON)
    np.fill_diagonal(A, 0.0)
    W = A / A.sum(axis=1, keepdims=True)
    return validate_W(W), {"trade_average_min": float(avg_trade.min()),
                           "trade_average_max": float(avg_trade.max()),
                           "income_group_counts": Counter(map(int, income)).most_common(),
                           "coordinate_standardization": bool(standardize)}


def validate_W(W):
    W = np.asarray(W, float)
    if W.shape != (37, 37):
        raise ValueError(f"Expected a 37x37 W, not {W.shape}")
    if not np.isfinite(W).all() or np.any(W < 0):
        raise ValueError("W must be finite and nonnegative")
    if not np.allclose(np.diag(W), 0, atol=1e-12):
        raise ValueError("W must have zero diagonal")
    if np.any(W.sum(axis=1) <= 0):
        raise ValueError("Every W row must have positive weight")
    # Respect the paper's row standardization even for supplied W.
    W = W / W.sum(axis=1, keepdims=True)
    assert np.allclose(W.sum(axis=1), 1)
    return W


def read_W_csv(path, countries):
    """CSV format: Country,<country1>,...,<country37> with 37 country rows.

    The column and row ordering is reordered to the analysis panel country list.
    """
    rows = load_rows(path)
    cols = set(rows[0]) - {"Country"}
    if cols != set(countries):
        raise ValueError(f"Weights CSV country columns differ; missing={sorted(set(countries)-cols)}, "
                         f"extra={sorted(cols-set(countries))}")
    mapping = {str(r["Country"]).strip(): r for r in rows}
    if set(mapping) != set(countries) or len(rows) != 37:
        raise ValueError("Weights CSV must have exactly 37 named country rows")
    return validate_W(np.array([
        [number(mapping[row], col, f"W({row},{col})") for col in countries]
        for row in countries
    ], float))


def save_W(path, W, countries):
    rows = [{"Country": countries[i], **{countries[j]: float(W[i, j]) for j in range(len(countries))}}
            for i in range(len(countries))]
    write_csv(path, rows, ["Country"] + countries)


def spatial_design(records, countries, W, include_year=True):
    """The paper's restricted Durbin spatial--Mundlak regressors + year FE.

    Spatial lag of DFI WITHIN only. Does not lag Gov or country means.
    Income_Level enters as time-invariant regressor, as in original Eq. (6).
    W is held constant across all sample years.
    """
    N = len(countries)
    within_by_year = np.array([
        [next(r["DFI_within"] for r in records if r["Country"] == c and r["Year"] == y)
         for c in countries] for y in YEARS
    ])
    WDFI = (within_by_year @ W.T).reshape(-1)  # year-major: (Wy_t)_i
    names = ["const", "W_ln_GDPpc_rho_regressor_ENDOGENOUS", "DFI_within", "W_DFI_within",
             "DFI_between", "Gov_within", "Gov_between", "Income_Level"]
    # The endogenous outcome lag is NOT entered in ML linear X; it is estimated
    # as rho in A(rho) y = X b + u.
    y = np.array([r["ln_GDPpc"] for r in records], float)
    Xnames = ["const", "DFI_within", "W_DFI_within", "DFI_between", "Gov_within",
              "Gov_between", "Income_Level"]
    if include_year:
        Xnames += [f"Year_{y}" for y in YEARS[1:]]
    X = np.array([
        [1, r["DFI_within"], WDFI[idx], r["DFI_between"], r["Gov_within"],
         r["Gov_between"], r["Income_Level"]]
        + ([float(r["Year"] == yr) for yr in YEARS[1:]] if include_year else [])
        for idx, r in enumerate(records)
    ], dtype=float)
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Spatial model matrix rank deficient (income classification/means?)")
    return y, X, Xnames


def spatial_ml(y, X, names, W, years=len(YEARS)):
    """Gaussian pooled-panel SAR ML with year effects in the design matrix.

    log L = T log |I-rho W| - n/2 log(2 pi sigma^2)
            - (1/(2 sigma^2)) ||(I_T kron (I-rho W)) y - X b||^2.
    Maximize the concentrated likelihood over rho, conditional on fixed W.
    Errors are assumed homoskedastic and independent for ML standard errors;
    the resulting SEs are *not* cluster-robust. Not causal identification.
    """
    N = W.shape[0]
    obs = y.size
    assert obs == N*years
    eye = np.eye(N)
    y2 = y.reshape((years, N))
    Wy = (y2 @ W.T).reshape(-1)
    pinvX = np.linalg.pinv(X)
    LOG2PI = math.log(2*math.pi)

    def prof(rho):
        B = eye - rho * W
        sign, ldet = np.linalg.slogdet(B)
        if sign <= 0 or not np.isfinite(ldet):
            return (-np.inf, None, None)
        ystar = y - rho * Wy
        bhat = pinvX @ ystar
        e = ystar - X @ bhat
        sig2 = float(np.dot(e, e) / obs)
        if sig2 <= 0:
            return (-np.inf, None, None)
        ll = years * ldet - 0.5 * obs * (LOG2PI + math.log(sig2) + 1)
        return (float(ll), bhat, sig2)

    # Grid scan + local optimizations reduces dependence on a single unimodal
    # assumption for the concentrated spatial likelihood in rho.
    grid = np.linspace(-0.98, .98, 99)
    scores = np.array([prof(float(r))[0] for r in grid])
    good = np.argsort(scores)[-6:]
    candidates = [float(grid[j]) for j in good]
    for j in good:
        lo = float(grid[max(0, j-1)])
        hi = float(grid[min(len(grid)-1, j+1)])
        fit = optimize.minimize_scalar(lambda rho: -prof(rho)[0], bounds=(lo, hi), method="bounded",
                                       options={"xatol": 1e-10})
        if fit.success:
            candidates.append(float(fit.x))
    rho = max(candidates, key=lambda r: prof(r)[0])
    ll, bhat, sig2 = prof(rho)
    if abs(rho) > .975:
        print("WARNING: rho is close to +/-0.98 bound; spatial estimate may be boundary-sensitive", file=sys.stderr)

    def negative_joint_likelihood(z):
        this_rho = float(z[0])
        if abs(this_rho) >= .999:
            return 1e80
        b = np.asarray(z[1:-1])
        log_s2 = float(z[-1])
        B = eye - this_rho * W
        sign, ldet = np.linalg.slogdet(B)
        if sign <= 0:
            return 1e80
        u = y - this_rho * Wy - X @ b
        return -(years * ldet - 0.5 * obs * (LOG2PI + log_s2)
                 - 0.5 * np.dot(u,u) * math.exp(-log_s2))

    zhat = np.r_[rho, bhat, math.log(sig2)]
    se = np.full(zhat.size, np.nan)
    variance = None
    se_note = "ML observed-Hessian normal approximation; assumes exogenous W, iid Gaussian errors"
    try:
        hessian = approx_hess(zhat, negative_joint_likelihood)
        eigen = np.linalg.eigvalsh(hessian)
        if np.min(eigen) <= 0 or np.linalg.cond(hessian) > 1e12:
            raise ArithmeticError("Hessian non-positive-definite or ill-conditioned")
        variance = np.linalg.inv(hessian)
        se = np.sqrt(np.diag(variance))
    except (ArithmeticError, np.linalg.LinAlgError) as e:
        se_note = f"Observed-Hessian standard errors unavailable: {e}"

    param_names = ["rho_W_ln_GDPpc"] + names + ["log_sigma2"]
    rows = []
    for j, (name, estimate) in enumerate(zip(param_names, zhat)):
        sigma = float(se[j]) if np.isfinite(se[j]) else None
        pval = float(2*stats.norm.sf(abs(estimate/sigma))) if sigma is not None and sigma > 0 else None
        rows.append({"term": name, "estimate": float(estimate), "std_error": sigma,
                     "p_value": pval,
                     "ci_low": float(estimate-1.96*sigma) if sigma is not None else None,
                     "ci_high": float(estimate+1.96*sigma) if sigma is not None else None})
    result = {"n": obs, "n_countries": N, "n_years": years,
              "log_likelihood": ll, "sigma2_mle": float(sig2), "rho": float(rho),
              "rho_grid_limit": .98, "standard_error_assumptions": se_note,
              "spatial_parameters": rows}
    return result, variance


def impacts(rho, beta, theta, gamma, W, covariance, param_names):
    """LeSage/Elhorst model implied average direct/indirect/total impacts.

    Average DFI multiplier: (I-rho W)^(-1) (beta I + theta W).
    For governance within: (I-rho W)^(-1) gamma I.
    Standard errors: delta method using Gaussian ML covariance only.
    """
    N = W.shape[0]
    eye = np.eye(N)

    def triplet(r, b, t):
        S = np.linalg.solve(eye - r*W, b*eye + t*W)
        direct = float(np.trace(S)/N)
        total = float(S.sum()/N)
        return np.array([direct, total-direct, total])

    output = []
    for name, par, keys in [
        ("DFI_within", [rho,beta,theta], ["rho_W_ln_GDPpc","DFI_within","W_DFI_within"]),
        ("Gov_within", [rho,gamma,0.0], ["rho_W_ln_GDPpc","Gov_within"]),
    ]:
        values = triplet(*par)
        J = None
        matrix = None
        if covariance is not None:
            indices = [param_names.index(k) for k in keys]
            matrix = covariance[np.ix_(indices,indices)]
            J = np.zeros((3,len(keys)))
            for j in range(len(keys)):
                pplus = np.array(par,float)
                pminus = np.array(par,float)
                h = 1e-5 * max(1, abs(par[j]))
                pplus[j] += h
                pminus[j] -= h
                J[:,j] = (triplet(*pplus)-triplet(*pminus))/(2*h)
        for j, impact_type in enumerate(("average_direct","average_indirect","average_total")):
            sigma = float(math.sqrt(max(0,float(J[j]@matrix@J[j])))) if J is not None else None
            value = float(values[j])
            output.append({"regressor": name, "impact": impact_type, "estimate": value,
                           "delta_se": sigma,
                           "p_value_normal": float(2*stats.norm.sf(abs(value/sigma))) if sigma is not None and sigma>0 else None,
                           "ci_low": value-1.96*sigma if sigma is not None else None,
                           "ci_high": value+1.96*sigma if sigma is not None else None})
    return output


def run_spatial(records, countries, W, out_dir):
    y, X, names = spatial_design(records, countries, W, include_year=True)
    result, covariance = spatial_ml(y, X, names, W)
    _, X0, names0 = spatial_design(records, countries, W, include_year=False)
    no_year, _ = spatial_ml(y, X0, names0, W)
    LR = max(0., 2*(result["log_likelihood"] - no_year["log_likelihood"]))
    result["year_fixed_effect_LR_test"] = {"LR_statistic": LR,"df": 6,
                                          "p_value_chi2": float(stats.chi2.sf(LR, 6)),
                                          "no_year_loglik": no_year["log_likelihood"]}
    write_csv(out_dir / "spatial_SML_yearFE_coefficients.csv", result["spatial_parameters"])
    param_names = [row["term"] for row in result["spatial_parameters"]]
    coef = {row["term"]:row["estimate"] for row in result["spatial_parameters"]}
    effects = impacts(coef["rho_W_ln_GDPpc"], coef["DFI_within"], coef["W_DFI_within"],
                      coef["Gov_within"], W, covariance, param_names)
    write_csv(out_dir / "spatial_SML_impacts.csv", effects)
    result["direct_indirect_total_impacts"] = effects
    result["weights_diagnostics"] = {"zero_diagonal": bool(np.allclose(np.diag(W),0)),
                                      "row_sums_approx_one": bool(np.allclose(W.sum(axis=1),1)),
                                      "largest_nonself_weight": float(W.max()),
                                      "median_row_largest_weight": float(np.median(W.max(axis=1))),
                                      "epsilon": EPSILON}
    return result


def markdown_report(out_path, records, countries, summary, regressions, spatial_result, notes):
    lines = ["# OIC Analysis — Reproduction Report",
             "", "**Status:** The core uploaded dataset reproduces the original manuscript's sample and descriptive statistics. \n"
             "Spatial estimation is valid only if original spatial weights or their source variables are supplied.",
             "", f"Countries: **{len(countries)}**; observations: **{len(records)}**; years: **2017–2023**.",
             "", "## Exact sample reconciliation", "",
             "The original workbook contains 40 countries for 2015–2024. The manuscript's 37 countries "
             "correspond to dropping Afghanistan, Libya, and Sudan; the seven-year window yields 259 observations.",
             "", "| Measure | Mean | SD (sample) | Min | Max |", "|---|---:|---:|---:|---:|"]
    for col, v in summary.items():
        lines.append(f"| {col} | {v['mean']:.4f} | {v['sd_sample']:.4f} | {v['min']:.4f} | {v['max']:.4f} |")
    lines += ["", "## Models estimated", "",
              "**Non-spatial checks (estimated):** two-way country/year fixed effects and year-FE "
              "Mundlak regressions, with country-clustered standard errors. These do NOT solve the "
              "endogenous spatial-outcome-lag problem in the original regression. Avoid replacing the "
              "manuscript's spatial table with these as though they were the spatial estimator.", ""]
    for model_name, vals in regressions.items():
        lines += [f"### {model_name}", "",
                  f"R² = {vals['r_squared']:.4f}, country clusters = {vals['cluster_count']}.", "",
                  "| Term | Estimate | Cluster SE | p-value |", "|---|---:|---:|---:|"]
        selected = [x for x in vals["coefficients"] if x["term"] in
                    {"DFI_within","DFI_between","Gov_within","Gov_between","Income_Level","DFI_Index","Gov"}]
        for row in selected:
            lines.append(f"| {row['term']} | {row['coefficient']:.5f} | {row['std_error']:.5f} | {row['p_value']:.4g} |")
        lines.append("")
    if spatial_result is None:
        lines += ["## Spatial maximum likelihood: NOT RUN", "",
                  "The uploaded workbook has no `Trade_Op_pct`, no baseline `Income_Level`, and "
                  "no 37×37 weight matrix. Equation (6) needs these to compute the paper's "
                  "`W_ln_GDPpc` and `W_DFI_within`. Do not substitute `ED`, `IF`, or `Edu` "
                  "without verified definitions and a revised research design.", "",
                  "Provide either (a) the original 37×37 W CSV, or (b) the two missing "
                  "columns in `trade_income_template.csv` to estimate the model as written.", ""]
    else:
        lines += ["## Spatial maximum likelihood: ESTIMATED", "",
                  "Restricted Durbin SAR ML, fixed W, within–between DFI/Gov, fixed income classification, "
                  "and year effects (2018–2023; 2017 reference). Homoskedastic Gaussian errors assumed.", "",
                  "| Term | ML estimate | ML SE | p-value |", "|---|---:|---:|---:|"]
        for c in spatial_result["spatial_parameters"]:
            if c["term"] in ("rho_W_ln_GDPpc", "DFI_within","W_DFI_within","DFI_between",
                             "Gov_within","Gov_between","Income_Level"):
                se = c["std_error"]
                pv = c["p_value"]
                lines.append(f"| {c['term']} | {c['estimate']:.5f} | {se:.5f} | {pv:.4g} |" if se is not None else
                             f"| {c['term']} | {c['estimate']:.5f} | unavailable | unavailable |")
        lines.extend(["", "### Average spatial impacts (conditional ML)", "",
                      "| Variable | Type | Estimate | ML delta-SE |", "|---|---|---:|---:|"])
        for v in spatial_result["direct_indirect_total_impacts"]:
            lines.append(f"| {v['regressor']} | {v['impact']} | {v['estimate']:.5f} | "
                         f"{v['delta_se']:.5f} |" if v['delta_se'] is not None else
                         f"| {v['regressor']} | {v['impact']} | {v['estimate']:.5f} | unavailable |")
        lr = spatial_result["year_fixed_effect_LR_test"]
        lines += ["",f"LR test of year indicators: χ²(6) = {lr['LR_statistic']:.3f}, "
                  f"p = {lr['p_value_chi2']:.4g} (under spatial ML assumptions).", ""]
    lines += ["## Necessary cautions and outstanding author inputs", "",
              "- The index is already present; its component series, exact normalization, "
              "weighting, rescaling, and validation are **not recoverable** from the final index values alone.",
              "- World Bank income/currency/price series codes and governance aggregation source are unverified.",
              "- The original D12 averaging period is not recorded. This code uses 2017–2023 "
              "for all Mundlak averages, and for trade average if supplied.",
              "- Original weights derived from economic similarity are not proof of exogeneity; "
              "spatial ML consistency requires stronger assumptions than merely using a different estimator.",
              "- The numerical model uses pooled spatial ML with year dummies and Mundlak terms; "
              "this is not an unrestricted country-effects model or a causal design.",
              "- Standard errors for nonspatial regressions are clustered by country. "
              "Spatial ML Hessian standard errors assume iid Gaussian disturbances and fixed W; "
              "they are not cluster-robust.",
              "- Re-estimated coefficients must replace original OLS values in the abstract, "
              "Tables 4/Hypotheses, discussion, and conclusion before claiming reviewer resolution.",
              "", "## Method notes", ""]
    lines.extend(f"- {msg}" for msg in notes)
    out_path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("OIC_Input_37x7.csv"),
                        help="Original country-year CSV (bundled) or workbook .xlsx")
    parser.add_argument("--supplement", type=Path,
                        help="Additional verified Country,Year,Trade_Op_pct,Income_Level CSV")
    parser.add_argument("--weights", type=Path,
                        help="Original 37x37 W matrix CSV, alternative to supplement trade/income for W")
    parser.add_argument("--mode", choices=["baseline", "spatial"], default="baseline")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--standardize-weights", action="store_true", help="SENSITIVITY ONLY: z-score weight coordinates")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    records, countries, original_countries = load_panel(args.data)
    have_trade, have_income = enrich_with_supplement(records, args.supplement)
    summary = descriptives(records)
    ols = ols_checks(records,countries,have_income,args.output)
    notes = ["Excluded Afghanistan, Libya, Sudan as shown by manuscript's 37-country list.",
             "All coefficients are associations, not identified causal effects."]
    spatial_result = None
    if args.mode == "spatial":
        if not have_income:
            parser.error("Spatial model requires original baseline Income_Level for all 37 countries. "
                         "Provide --supplement (it cannot be inferred from GDPpc without redefining the paper).")
        if args.weights:
            W = read_W_csv(args.weights, countries)
            notes.append("W was read from externally supplied, named and validated 37x37 matrix.")
        elif have_trade:
            W, wmeta = build_W_from_trade(records,countries,standardize=args.standardize_weights)
            notes.append("W built using inverse Euclidean economic-similarity distance and 1e-5 regularizer; "
                         "coordinate standardized for sensitivity only" if args.standardize_weights else
                         "W built from raw trade-openness percentage and binary income-group coordinates per original manuscript.")
            write_json(args.output/"weight_construction.json",wmeta)
        else:
            parser.error("Spatial model requires original W via --weights, or Trade_Op_pct via --supplement")
        save_W(args.output/"W_matrix_37x37.csv",W,countries)
        spatial_result = run_spatial(records,countries,W,args.output)
        write_json(args.output/"spatial_ML_results.json",spatial_result)
    write_json(args.output/"data_and_nonspatial_results.json", {
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "original_country_count":original_countries,
        "excluded_countries":sorted(EXCLUDED), "estimation_years":list(YEARS),
        "country_list":countries,"n_countries":len(countries), "n":len(records),
        "Mundlak_average_years":list(YEARS),
        "descriptives": summary,"nonspatial_checks":ols,
        "spatial_mle_completed":spatial_result is not None,
    })
    write_csv(args.output/"sample_37x7.csv",records)
    markdown_report(args.output/"ANALYSIS_REPORT.md", records,countries,summary,ols,spatial_result,notes)
    print(f"Validated: 37 countries, 2017-2023, 259 observations")
    print("DFI_Index mean:",round(summary['DFI_Index']['mean'],4))
    print("Governance mean:",round(summary['Gov']['mean'],4))
    print("ln_GDPpc mean:",round(summary['ln_GDPpc']['mean'],4))
    print("Non-spatial clustered year-FE regressions: COMPLETED")
    print("Spatial SML:","COMPLETED" if spatial_result else "NOT RUN (original W / source variables missing)")
    print(f"Report: {args.output/'ANALYSIS_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
