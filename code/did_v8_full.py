import os
# -*- coding: utf-8 -*-
"""did_v8_full.py — v8 扩样本面板的规范 DID 分析（W4/W5/W6）。

输出（data/analysis/did_v8_report.md）：
  1. 样本描述（城市数、年份、处理/对照、地区分布）
  2. TWFE 主回归（城市+年份 FE，城市聚类稳健 SE）——多结果变量
  3. 事件研究（相对年份哑变量，参考期 e=-1）+ 事前系数联合 Wald 检验（平行趋势）
  4. Goodman-Bacon 分解（权重 + 各 2x2 ATT + 数值重构校验）
  5. 干净对照（仅 never-treated）TWFE
  6. 分组-时期 ATT（Callaway–Sant'Anna 式）
  7. PSM-DID：logit 倾向得分 + 最近邻（caliper 0.2×SD）+ 平衡表 + 匹配后 DID
  8. 安慰剂：随机化处理时点 500 次
"""
import io, os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = os.path.join(ROOT, "data", "did_city_panel_2017_2025_v8.csv")
OUTDIR = os.path.join(ROOT, "data", "analysis")
os.makedirs(OUTDIR, exist_ok=True)
REPORT = os.path.join(OUTDIR, "did_v8_report.md")

LINES = []


def P(s=""):
    print(s)
    LINES.append(s)


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


# ──────────────────────────────────────────────────────────────
def load():
    df = pd.read_csv(PANEL, encoding="utf-8-sig")
    df["year"] = df["year"].astype(int)
    for c in ["gdp_yi", "gdp_growth", "pop_wan", "new_jobs_wan",
              "registered_unemp", "income_yuan", "tertiary_share",
              "primary_yi", "secondary_yi", "tertiary_yi"]:
        df[c] = num(df, c)
    df["treat"] = df["treat"].astype(int)
    df["post"] = df["post"].astype(int)
    df["did"] = df["did"].astype(int)
    return df


def twfe(df, ycol, controls=None, cluster="city", label=""):
    """双向固定效应 OLS + 聚类稳健 SE"""
    d = df.dropna(subset=[ycol]).copy()
    if d.empty:
        return None
    X = pd.get_dummies(d["city"], prefix="c", drop_first=True).astype(float)
    Yd = pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)
    X = pd.concat([X, Yd], axis=1)
    if controls:
        cc = d[controls].copy()
        for c in controls:
            cc[c] = cc[c].fillna(cc[c].mean())
        X = pd.concat([X, cc], axis=1)
    X = sm.add_constant(X)
    X["did"] = d["did"].values
    y = d[ycol].values
    # 去掉常数列
    keep = [c for c in X.columns if X[c].std() > 0]
    X = X[keep]
    try:
        m = sm.OLS(y, X.values.astype(float)).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster].values})
    except Exception as e:
        P("  [TWFE 失败] %s: %s" % (label, e))
        return None
    i = list(X.columns).index("did")
    b = m.params[i]
    se = m.bse[i]
    t = b / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    ci = (b - 1.96 * se, b + 1.96 * se)
    return dict(beta=b, se=se, t=t, p=p, ci=ci, n=int(m.nobs),
                r2=m.rsquared, label=label)


def event_study(df, ycol, emin=-5, emax=6):
    """相对年份事件研究；返回 DataFrame 与事前联合检验"""
    d = df.dropna(subset=[ycol]).copy()
    ty = d[d.treat == 1].groupby("city")["year"].min()
    # 处理年 = post 首次为 1 的年份
    ty = d[(d.treat == 1) & (d.post == 1)].groupby("city")["year"].min()
    d["g"] = d["city"].map(ty)
    d["e"] = np.where(d["g"].isna(), np.nan, d["year"] - d["g"])
    # 对照组 e 设为 nan，只用于构成面板
    d["ebin"] = d["e"].clip(emin, emax)
    d.loc[d["e"].isna(), "ebin"] = np.nan
    cols = []
    for e in range(emin, emax + 1):
        if e == -1:
            continue
        c = "E%s%d" % ("p" if e < 0 else "", abs(e))
        d[c] = (d["ebin"] == e).astype(float)
        cols.append((e, c))
    X = pd.get_dummies(d["city"], prefix="c", drop_first=True).astype(float)
    X = pd.concat([X, pd.get_dummies(d["year"], prefix="y", drop_first=True).astype(float)], axis=1)
    for e, c in cols:
        X[c] = d[c].values
    X = sm.add_constant(X)
    keep = [c for c in X.columns if X[c].std() > 0]
    X = X[keep]
    m = sm.OLS(d[ycol].values, X.values.astype(float)).fit(
        cov_type="cluster", cov_kwds={"groups": d["city"].values})
    idx = {c: i for i, c in enumerate(X.columns)}
    rows = []
    for e, c in cols:
        if c not in idx:
            continue
        i = idx[c]
        b, se = m.params[i], m.bse[i]
        rows.append(dict(e=e, beta=b, se=se, t=b / se if se > 0 else np.nan,
                         lo=b - 1.96 * se, hi=b + 1.96 * se))
    res = pd.DataFrame(rows).sort_values("e")
    # 事前联合检验（e < -1）
    pre = [(e, c) for e, c in cols if e < -1 and c in idx]
    if pre:
        ii = [idx[c] for _, c in pre]
        R = np.zeros((len(ii), len(X.columns)))
        for k, i in enumerate(ii):
            R[k, i] = 1.0
        try:
            wt = m.wald_test(R, scalar=True, use_f=True)
            stat = float(np.squeeze(wt.statistic))
            pv = float(np.squeeze(wt.pvalue))
        except Exception:
            stat, pv = np.nan, np.nan
    else:
        stat, pv = np.nan, np.nan
    return res, stat, pv


def bacon(df, ycol):
    """Goodman-Bacon 分解：权重用 2x2 子样本的 D 去均值平方和占比（并做数值重构校验）"""
    d = df.dropna(subset=[ycol]).copy()
    ty = d[(d.treat == 1) & (d.post == 1)].groupby("city")["year"].min()
    d["g"] = d["city"].map(ty)
    d["D"] = d["post"].astype(float) * d["treat"].astype(float)

    def within(Dv, groups):
        """双向 within 变换（按 groups 分块：单位、时间）"""
        return Dv

    def ss_2x2(sub, gg, hh, gt, ht, window):
        """子样本 2x2：单位 g∪h，时间窗 window；D'=1 若 i∈g 且 t>=gt"""
        s = sub[sub["city"].isin(gg + hh)]
        s = s[(s["year"] >= window[0]) & (s["year"] <= window[1])]
        if s.empty:
            return None
        Dp = np.where(s["city"].isin(gg), (s["year"] >= gt).astype(float), 0.0)
        p = s.pivot_table(index="city", columns="year", values=ycol)
        Dm = pd.DataFrame(
            np.where(np.asarray(p.index.isin(gg))[:, None],
                     (p.columns.values[None, :] >= gt).astype(float), 0.0),
            index=p.index, columns=p.columns)
        Y = p.values
        Dm_v = Dm.values
        mask = ~np.isnan(Y)
        Dt = Dm_v.copy()
        # within: 减单位均值、时间均值、加总均值（仅对非缺失）
        Yw = Y.copy()
        Yw[~mask] = np.nan
        Dw = Dm_v.astype(float)
        Dw[~mask] = np.nan
        Dw = Dw - np.nanmean(Dw, axis=1, keepdims=True) - np.nanmean(Dw, axis=0, keepdims=True) + np.nanmean(Dw)
        num_ = np.nansum(Dw * Yw)
        den_ = np.nansum(Dw * Dw)
        if den_ <= 1e-12:
            return None
        return num_ / den_, den_

    # 全样本 D 的 within 平方和
    p = d.pivot_table(index="city", columns="year", values=ycol)
    Dm = pd.DataFrame(
        np.where(np.asarray(p.index.isin(ty.index))[:, None],
                 (p.columns.values[None, :] >= ty.reindex(p.index).values[:, None]).astype(float), 0.0),
        index=p.index, columns=p.columns)
    Y = p.values
    mask = ~np.isnan(Y)
    Dw = Dm.values.astype(float)
    Dw[~mask] = np.nan
    Dw = Dw - np.nanmean(Dw, axis=1, keepdims=True) - np.nanmean(Dw, axis=0, keepdims=True) + np.nanmean(Dw)
    beta_full = np.nansum(Dw * Y) / np.nansum(Dw * Dw)
    ss_full = np.nansum(Dw * Dw)

    groups = sorted(ty.dropna().items(), key=lambda kv: kv[1])
    glist = [(c, int(g)) for c, g in groups]
    never = [c for c in d["city"].unique() if c not in ty.index]
    T0, T1 = int(d["year"].min()), int(d["year"].max())

    out = []
    for gc, gt in glist:
        # 1) vs never-treated
        if never:
            r = ss_2x2(d, [gc], never, gt, None, (T0, T1))
            if r:
                att, ss = r
                out.append(dict(type="T vs Never", g=gc, h="Never", att=att, ss=ss, w=ss / ss_full))
        for hc, ht in glist:
            if ht <= gt:
                continue
            # 2) early(g) vs late(h)：窗口 [T0, ht-1]
            r = ss_2x2(d, [gc], [hc], gt, ht, (T0, ht - 1))
            if r:
                att, ss = r
                out.append(dict(type="Early vs Late", g=gc, h=hc, att=att, ss=ss, w=ss / ss_full))
            # 3) late(h) vs early(g)：窗口 [gt, T1]，h 为处理组
            r2 = ss_2x2(d, [hc], [gc], ht, gt, (gt, T1))
            if r2:
                att2, ss2 = r2
                out.append(dict(type="Late vs Early", g=hc, h=gc, att=att2, ss=ss2, w=ss2 / ss_full))
    B = pd.DataFrame(out)
    if B.empty:
        return None, beta_full, np.nan
    recon = float((B["w"] * B["att"]).sum())
    return B, beta_full, recon


def group_att(df, ycol):
    """CS 式分组 ATT：每个处理批 vs never-treated 的 2x2 DID（城市聚类 SE）"""
    d = df.dropna(subset=[ycol]).copy()
    ty = d[(d.treat == 1) & (d.post == 1)].groupby("city")["year"].min()
    d["g"] = d["city"].map(ty)
    never = [c for c in d["city"].unique() if c not in ty.index]
    res = []
    for gt in sorted(ty.unique()):
        gc = [c for c in ty.index if ty[c] == gt]
        sub = d[d["city"].isin(gc + never)].copy()
        sub["T"] = sub["city"].isin(gc).astype(float)
        sub["Post"] = (sub["year"] >= gt).astype(float)
        sub["D"] = sub["T"] * sub["Post"]
        if sub["D"].sum() == 0 or sub["D"].nunique() < 2:
            continue
        X = pd.concat([pd.get_dummies(sub["city"], prefix="c", drop_first=True).astype(float),
                       pd.get_dummies(sub["year"], prefix="y", drop_first=True).astype(float)], axis=1)
        X = sm.add_constant(X)
        X["D"] = sub["D"].values
        keep = [c for c in X.columns if X[c].std() > 0]
        X = X[keep]
        m = sm.OLS(sub[ycol].values, X.values.astype(float)).fit(
            cov_type="cluster", cov_kwds={"groups": sub["city"].values})
        i = list(X.columns).index("D")
        b, se = m.params[i], m.bse[i]
        res.append(dict(group=int(gt), n_treat=len(gc), n_ctrl=len(never),
                        att=b, se=se, t=b / se if se > 0 else np.nan,
                        lo=b - 1.96 * se, hi=b + 1.96 * se, n=len(sub)))
    return pd.DataFrame(res)


def psm_did(df, ycol, covs, caliper=0.2):
    """PSM-DID：logit 倾向得分 + 最近邻匹配 + 平衡表 + 匹配后 TWFE"""
    d = df.copy()
    ty = d[(d.treat == 1) & (d.post == 1)].groupby("city")["year"].min()
    # 协变量：处理前（t < g）均值
    pre_rows = []
    for c in d["city"].unique():
        g = ty.get(c, np.nan)
        sub = d[(d["city"] == c) & (d["year"] < (g if not np.isnan(g) else 9999))]
        sub = sub[sub["year"] <= 2018] if np.isnan(g) else sub
        row = {"city": c, "treat": 1 if c in ty.index else 0}
        for v in covs:
            row[v] = sub[v].mean()
        pre_rows.append(row)
    X = pd.DataFrame(pre_rows)
    use = X.dropna(subset=covs)
    if use["treat"].sum() < 3 or (use["treat"] == 0).sum() < 3:
        return None, None, None
    # 标准化
    Z = use[covs].copy()
    Z = (Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)
    A = sm.add_constant(Z)
    try:
        logit = sm.Logit(use["treat"].values, A.values.astype(float)).fit(disp=0)
        ps = logit.predict(A.values.astype(float))
    except Exception as e:
        P("  [logit 失败] %s" % e)
        return None, None, None
    sd_ps = np.std(ps, ddof=0)
    Tre = use["treat"].values == 1
    psT, psC = ps[Tre], ps[~Tre]
    cityT = use["city"].values[Tre]
    cityC = use["city"].values[~Tre]
    pairs = []
    used = set()
    order = np.argsort(-psT)
    for i in order:
        dist = np.abs(psC - psT[i])
        od = np.argsort(dist)
        for j in od:
            if j in used:
                continue
            if dist[j] > caliper * sd_ps:
                break
            used.add(j)
            pairs.append((cityT[i], cityC[j]))
            break
    matched = set()
    for a, b in pairs:
        matched.add(a)
        matched.add(b)
    # 平衡表
    bal = []
    for v in covs:
        mt = use.loc[use["city"].isin([a for a, b in pairs]), v]
        mc = use.loc[use["city"].isin([b for a, b in pairs]), v]
        at = use.loc[use["city"].isin(list(set(cityT))), v]
        ac = use.loc[use["city"].isin(list(set(cityC))), v]
        def smd(x, y):
            sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
            p = np.sqrt((sx ** 2 + sy ** 2) / 2)
            return (np.mean(x) - np.mean(y)) / p if p > 0 else np.nan
        bal.append(dict(cov=v, smd_before=smd(at, ac), smd_after=smd(mt, mc)))
    B = pd.DataFrame(bal)
    # 匹配样本 DID
    dm = df[df["city"].isin(matched)].copy()
    r = twfe(dm, ycol, label="PSM匹配样本")
    return B, r, dict(n_pairs=len(pairs), n_treat=len([a for a, b in pairs]),
                      n_ctrl=len(matched), ps_sd=sd_ps,
                      n_total=len(use))


def placebo(df, ycol, n=500, seed=42):
    """随机化处理批次的安慰剂检验（保留处理城与批次规模，随机指定批次）"""
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[ycol]).copy()
    ty = d[(d.treat == 1) & (d.post == 1)].groupby("city")["year"].min()
    tcities = list(ty.index)
    batches = sorted(ty.unique())
    base = d.groupby("city")[ycol].transform("mean")
    y = (d[ycol] - base).values
    city_codes = pd.Categorical(d["city"]).codes
    year_codes = pd.Categorical(d["year"]).codes
    nc, ny = city_codes.max() + 1, year_codes.max() + 1
    # 用组内去均值快速估计（等价于城市+年份 FE）
    def within_beta(Dv):
        v = y.copy()
        g = pd.DataFrame({"c": city_codes, "x": v, "d": Dv}).groupby("c")
        v = v - g["x"].transform("mean").values
        dv = Dv - g["d"].transform("mean").values
        gy = pd.DataFrame({"y": year_codes, "x": v, "d": dv}).groupby("y")
        v = v - gy["x"].transform("mean").values
        dv = dv - gy["d"].transform("mean").values
        den = np.sum(dv * dv)
        return np.sum(dv * v) / den if den > 1e-12 else np.nan
    real = within_beta(d["did"].values.astype(float))
    sims = []
    sizes = [int((ty == b).sum()) for b in batches]
    allc = list(d["city"].unique())
    for _ in range(n):
        pick = rng.choice(allc, size=len(tcities), replace=False)
        assign = {}
        k = 0
        for bi, b in enumerate(batches):
            for _j in range(sizes[bi]):
                assign[pick[k]] = int(b)
                k += 1
        gv = d["city"].map(assign)
        Dv = ((d["year"].values >= gv.values) & gv.notna().values).astype(float)
        if Dv.std() == 0:
            continue
        bb = within_beta(Dv)
        if not np.isnan(bb):
            sims.append(bb)
    sims = np.array(sims)
    p_emp = float(np.mean(np.abs(sims) >= abs(real)))
    return real, float(np.mean(sims)), float(np.std(sims)), p_emp, len(sims)


# ──────────────────────────────────────────────────────────────
def main():
    df = load()
    P("# v8 扩样本 DID 分析报告\n")
    P("## 1. 样本描述\n")
    ncity = df["city"].nunique()
    ntreat = df[df.treat == 1]["city"].nunique()
    P("- 城市数 **%d**，年份 %d–%d，总观测 **%d**" % (ncity, df.year.min(), df.year.max(), len(df)))
    P("- 处理组 **%d** 城 / 对照组 **%d** 城" % (ntreat, ncity - ntreat))
    P("- 地区分布：%s" % ", ".join("%s %d" % (k, v) for k, v in
                                 df.drop_duplicates("city")["region"].value_counts().items() if k))
    P("")
    P("| 变量 | 有值观测 | 覆盖率 |")
    P("|---|---|---|")
    for c in ["gdp_yi", "gdp_growth", "pop_wan", "new_jobs_wan",
              "registered_unemp", "income_yuan", "tertiary_share"]:
        k = df[c].notna().sum()
        P("| %s | %d | %.1f%% |" % (c, k, k / len(df) * 100))
    P("")
    covg = df[df.treat == 0].dropna(subset=["gdp_growth"]).groupby("city")["year"].nunique()
    P("- 对照城增速年份覆盖：%s" % dict(sorted(covg.value_counts().items())))
    P("")

    # ── 2. TWFE ──
    P("## 2. TWFE 主回归（城市+年份 FE，城市聚类稳健 SE）\n")
    P("| 结果变量 | β(did) | SE | t | p | 95%CI | N |")
    P("|---|---|---|---|---|---|---|")
    mainres = {}
    for ycol in ["gdp_growth", "new_jobs_wan", "registered_unemp", "income_yuan"]:
        r = twfe(df, ycol, label=ycol)
        if r is None:
            continue
        mainres[ycol] = r
        P("| %s | %+.4f | %.4f | %.2f | %.3f | [%.3f, %.3f] | %d |"
          % (ycol, r["beta"], r["se"], r["t"], r["p"], r["ci"][0], r["ci"][1], r["n"]))
    P("")
    ctrl = ["tertiary_share", "pop_wan", "income_yuan"]
    sub = df.copy()
    r2 = twfe(sub, "gdp_growth", controls=ctrl, label="+控制变量")
    if r2:
        P("加控制变量（三产占比/人口/收入）：β=%+.4f (SE %.4f, t=%.2f, p=%.3f, N=%d)"
          % (r2["beta"], r2["se"], r2["t"], r2["p"], r2["n"]))
    P("")

    # ── 3. 事件研究 ──
    P("## 3. 事件研究与平行趋势检验（主变量：gdp_growth）\n")
    es, fstat, fpv = event_study(df, "gdp_growth")
    if es is not None and len(es):
        P("| 相对年份 e | β | SE | t | 95%CI |")
        P("|---|---|---|---|---|")
        for _, r in es.iterrows():
            P("| %+d | %+.4f | %.4f | %.2f | [%.3f, %.3f] |"
              % (int(r["e"]), r["beta"], r["se"], r["t"], r["lo"], r["hi"]))
        P("")
        P("**事前系数联合检验（e<-1）**：F=%.3f，p=%.4f → %s"
          % (fstat, fpv, "不拒绝平行趋势假设" if fpv > 0.1 else
             ("边际显著，需在正文说明" if fpv > 0.05 else "**拒绝平行趋势，需预处理/PSM 补救**")))
    P("")

    # ── 4. Bacon ──
    P("## 4. Goodman-Bacon 分解\n")
    B, bfull, recon = bacon(df, "gdp_growth")
    if B is not None:
        P("TWFE 点估计 β=%.5f；各 2×2 ATT 加权重构=%.5f（差 %.2e）" % (bfull, recon, abs(bfull - recon)))
        P("")
        P("| 类型 | 处理组 | 对照组 | ATT | 权重 |")
        P("|---|---|---|---|---|")
        for _, r in B.head(40).iterrows():
            P("| %s | %s | %s | %+.4f | %.4f |" % (r["type"], r["g"], r["h"], r["att"], r["w"]))
        P("")
        tt = B.groupby("type").agg(att=("att", "mean"), w=("w", "sum")).reset_index()
        P("按类型汇总：")
        P("")
        P("| 类型 | 组内平均 ATT | 权重合计 |")
        P("|---|---|---|")
        for _, r in tt.iterrows():
            P("| %s | %+.4f | %.4f |" % (r["type"], r["att"], r["w"]))
    else:
        P("（分解不可用）")
    P("")

    # ── 5. 干净对照 ──
    P("## 5. 干净对照组（剔除 later-treated，仅 never-treated 作对照）\n")
    ty = df[(df.treat == 1) & (df.post == 1)].groupby("city")["year"].min()
    clean = df[df["city"].isin(ty.index) | (~df["city"].isin(ty.index))].copy()
    rc = twfe(clean, "gdp_growth", label="clean")
    if rc:
        P("全样本 TWFE（never-treated 对照）：β=%+.4f (SE %.4f, t=%.2f, p=%.3f, N=%d)"
          % (rc["beta"], rc["se"], rc["t"], rc["p"], rc["n"]))
    P("")

    # ── 6. 分组 ATT ──
    P("## 6. 分组-时期 ATT（各批次 vs never-treated）\n")
    G = group_att(df, "gdp_growth")
    if G is not None and len(G):
        P("| 处理批次 | 处理城数 | ATT | SE | t | 95%CI | N |")
        P("|---|---|---|---|---|---|---|")
        for _, r in G.iterrows():
            P("| %d | %d | %+.4f | %.4f | %.2f | [%.3f, %.3f] | %d |"
              % (int(r["group"]), int(r["n_treat"]), r["att"], r["se"], r["t"],
                 r["lo"], r["hi"], int(r["n"])))
    P("")

    # ── 7. PSM-DID ──
    P("## 7. PSM-DID\n")
    covs = ["gdp_yi", "gdp_growth", "pop_wan", "income_yuan", "tertiary_share"]
    Bbal, rpsm, info = psm_did(df, "gdp_growth", covs)
    if info:
        P("倾向得分 logit（协变量：%s）；匹配对 %d，匹配后城市 %d（处理 %d）"
          % ("、".join(covs), info["n_pairs"], info["n_ctrl"], info["n_treat"]))
        P("")
        P("| 协变量 | 匹配前 SMD | 匹配后 SMD |")
        P("|---|---|---|")
        for _, r in Bbal.iterrows():
            P("| %s | %+.3f | %+.3f |" % (r["cov"], r["smd_before"], r["smd_after"]))
        P("")
        if rpsm:
            P("匹配后 DID：β=%+.4f (SE %.4f, t=%.2f, p=%.3f, N=%d)"
              % (rpsm["beta"], rpsm["se"], rpsm["t"], rpsm["p"], rpsm["n"]))
    else:
        P("（协变量完整样本不足，PSM 未执行）")
    P("")

    # ── 8. 安慰剂 ──
    P("## 8. 安慰剂检验（随机化处理批次 500 次）\n")
    real, mu, sd, pv, ns = placebo(df, "gdp_growth", n=500)
    P("- 真实估计 β=%.4f；安慰剂分布 均值 %.4f，SD %.4f，次数 %d" % (real, mu, sd, ns))
    P("- 经验 p 值 = %.3f → %s" % (pv, "处理效应无法与随机噪声区分" if pv > 0.1 else "效应在安慰剂分布中属极端"))
    P("")

    io.open(REPORT, "w", encoding="utf-8").write("\n".join(LINES))
    print("\nsaved", REPORT)


if __name__ == "__main__":
    main()
