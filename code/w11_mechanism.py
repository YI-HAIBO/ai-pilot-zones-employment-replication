import os
# -*- coding: utf-8 -*-
"""
W11: 机制检验正名 —— 用 v8 面板（295 城）做正式中介 + 调节分析
替代旧的 23 城描述性 Pearson r（0.387/0.795）。

A. 中介：Baron-Kenny 三步 + Sobel 检验 + 城市聚类 bootstrap 间接效应 CI
   Y  = new_jobs_wan / gdp_growth / registered_unemp
   M  = tertiary_share（产业结构升级渠道）、income_yuan（收入/生产率渠道）
B. 调节：政策前(2017-2019)服务业占比分组异质性；招聘暴露度（W9）调节（附时点声明）

输出：data/analysis/w11_mechanism_report.md
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import io

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = R + r'\data\did_city_panel_2017_2025_v8.csv'
EXPO = R + r'\data\analysis\jobs_city_exposure.csv'
OUT = R + r'\data\analysis\w11_mechanism_report.md'


def num(df, c):
    return pd.to_numeric(df[c], errors='coerce')


def load():
    df = pd.read_csv(PANEL, encoding='utf-8-sig')
    df['year'] = df['year'].astype(int)
    for c in ['gdp_yi', 'gdp_growth', 'pop_wan', 'new_jobs_wan',
              'registered_unemp', 'income_yuan', 'tertiary_share',
              'primary_yi', 'secondary_yi', 'tertiary_yi']:
        df[c] = num(df, c)
    for c in ['treat', 'post', 'did']:
        df[c] = df[c].astype(int)
    return df


def twfe2(df, ycol, xcols, cluster='city'):
    """双向固定效应 OLS（城市+年份哑变量），返回 (params, cov, n)。
    xcols 为核心变量列表；返回其系数与协方差子块（按 xcols 顺序排列）。"""
    need = [ycol] + xcols
    d = df.dropna(subset=need).copy().reset_index(drop=True)
    if len(d) < 50:
        return None
    X = pd.get_dummies(d['city'], prefix='c', drop_first=True).astype(float)
    Yd = pd.get_dummies(d['year'], prefix='y', drop_first=True).astype(float)
    X = pd.concat([X, Yd, d[xcols].astype(float)], axis=1)
    X = sm.add_constant(X)
    keep = [c for c in X.columns if X[c].std() > 0]
    Xv = X[keep].values.astype(float)
    y = d[ycol].astype(float).values
    m = sm.OLS(y, Xv).fit(cov_type='cluster',
                          cov_kwds={'groups': d[cluster].values})
    kl = list(keep)
    pos = {c: kl.index(c) for c in xcols if c in kl}
    params = {c: m.params[pos[c]] for c in xcols if c in pos}
    sel = [pos[c] for c in xcols if c in pos]
    sub_cov = m.cov_params()[np.ix_(sel, sel)]
    return params, sub_cov, int(m.nobs)


def pz(v, s):
    t = v / s if s > 0 else 0.0
    return 2 * (1 - stats.norm.cdf(abs(t)))


def sobel(a, sa, b, sb):
    ab = a * b
    se = (a ** 2 * sb ** 2 + b ** 2 * sa ** 2) ** 0.5
    z = ab / se if se > 0 else np.nan
    return ab, se, z, 2 * (1 - stats.norm.cdf(abs(z)))


def asel(r, c):
    """从 twfe2 结果取 (beta, se)"""
    i = list(r[0]).index(c)
    return r[0][c], np.sqrt(r[1][i, i])


def _ols_cluster(X, y, groups):
    """裸 numpy OLS + 城市聚类稳健 SE（bootstrap 专用，追求速度）"""
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return None, None
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Xe = X * e[:, None]
    S = pd.DataFrame(Xe).groupby(groups).sum().values
    meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


def boot_indirect(df, ycol, mcol, n_boot=300):
    """城市层重抽样的 indirect effect a·b bootstrap（百分位 CI）。
    用预计算城市/年份哑变量矩阵 + 裸 numpy 回归提速。"""
    need = [ycol, mcol, 'did', 'city', 'year']
    d = df.dropna(subset=[ycol, mcol]).reset_index(drop=True)
    if len(d) < 50:
        return None
    D = pd.get_dummies(d['city']).astype(np.float64)
    Yd = pd.get_dummies(d['year']).astype(np.float64)
    base = np.hstack([D.values, Yd.values])
    core = np.column_stack([d['did'].astype(float).values,
                            d[mcol].astype(float).values])
    yv = d[ycol].astype(float).values
    mv = d[mcol].astype(float).values
    groups = d['city'].values
    cities = sorted(d['city'].unique())
    idx = {c: np.where(groups == c)[0] for c in cities}
    rng = np.random.default_rng(20261030)
    abs_ = []
    mi_pos = 1  # core 列：[did, mcol]
    for _ in range(n_boot):
        pick = rng.choice(cities, size=len(cities), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        b = base[rows]
        keep_cols = b.std(axis=0) > 0
        X1 = np.hstack([b[:, keep_cols], core[np.ix_(rows, [0])]])  # a: M ~ did
        X2 = np.hstack([b[:, keep_cols], core[rows]])               # c′: Y ~ did + M
        b1, s1 = _ols_cluster(X1, mv[rows], groups[rows])   # a: did → M
        b2, s2 = _ols_cluster(X2, yv[rows], groups[rows])   # b/c′: Y ~ did + M
        if b1 is None or b2 is None:
            continue
        a = b1[-1]           # did → M
        bcoef = b2[-1]       # M → Y
        ab = a * bcoef
        if np.isfinite(ab):
            abs_.append(ab)
    if len(abs_) < 100:
        return None
    lo, hi = np.percentile(abs_, [2.5, 97.5])
    return float(np.mean(abs_)), float(lo), float(hi), len(abs_)


df = load()
L = []
P = L.append

P('# W11 机制检验报告：中介效应与调节效应（v8 面板，295 城）')
P('')
P('**产出日期**：2026-10-30')
P('**目的**：把旧的 23 城描述性相关（r=0.387/0.795）升级为面板内正式机制检验，')
P('支撑路径 A 叙事：「AI 试验区提升生产率与收入、推动产业结构升级，')
P('但就业总量未同步扩张 → 结构性就业风险」。')
P('')
P('## 一、中介检验设计（Baron-Kenny + Sobel）')
P('')
P('| 路径 | 含义 |')
P('|---|---|')
P('| a: did → M | 政策对中介变量的效应（双向固定效应） |')
P('| b: M → Y | 控制政策后中介对结果变量的效应 |')
P('| c: did → Y | 总效应；c′: did → Y（含 M 后）的直接效应 |')
P('| a·b | 间接效应，Sobel z 检验 + 300 次城市层 bootstrap CI |')
P('')
P('所有回归含城市与年份固定效应、城市聚类稳健 SE；')
P('M 为时间变动变量（产业结构、收入水平），在面板内可被双向 FE 识别。')
P('')


def mediation_table(mcol, mname):
    rows = []
    for ycol in ['new_jobs_wan', 'gdp_growth', 'registered_unemp']:
        tot = twfe2(df, ycol, ['did'])
        a = twfe2(df, mcol, ['did'])
        both = twfe2(df, ycol, ['did', mcol])
        if not (tot and a and both):
            continue
        c, sc = asel(tot, 'did')
        av, sa = asel(a, 'did')
        bv, sb = asel(both, mcol)
        cp, scp = asel(both, 'did')
        ab, se, z, pv = sobel(av, sa, bv, sb)
        rows.append(dict(y=ycol, c=c, pc=pz(c, sc), n=tot[2],
                         a=av, pa=pz(av, sa),
                         b=bv, pb=pz(bv, sb),
                         cp=cp, pcp=pz(cp, scp), ab=ab, z=z, pv=pv))
    P('## %s：中介结果（M = %s）' % (mname, mcol))
    P('')
    P('| 结果 Y | 总效应 c (p) | a: did→M (p) | b: M→Y (p) | c′ 直接效应 (p) | 间接 a·b | Sobel z | p | N |')
    P('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        P('| %s | %+.3f (%.3f) | %+.3f (%.3f) | %+.3f (%.3f) | %+.3f (%.3f) | %+.4f | %.2f | %.3f | %d |' % (
            r['y'], r['c'], r['pc'], r['a'], r['pa'], r['b'], r['pb'],
            r['cp'], r['pcp'], r['ab'], r['z'], r['pv'], r['n']))
    P('')
    for ycol in ['new_jobs_wan', 'gdp_growth']:
        r = boot_indirect(df, ycol, mcol)
        if r:
            P('- %s 间接效应 bootstrap(300 次)：均值 %+.4f，95%% CI [%+.4f, %+.4f]（有效重复 %d）'
              % (ycol, r[0], r[1], r[2], r[3]))
    P('')
    return rows


rows_A = mediation_table('tertiary_share', '二、产业结构升级渠道')
rows_B = mediation_table('income_yuan', '三、收入/生产率渠道')

# ───────── 调节 1：政策前服务业占比 ─────────
pre = df[df['year'] <= 2019].groupby('city')['tertiary_share'].mean()
med = pre.median()
df['pre_high'] = df['city'].map((pre > med).astype(int)).fillna(0).astype(int)
df['did_x_high'] = df['did'] * df['pre_high']

P('## 四、调节效应 1：政策前服务业占比（2017-2019 三产均值，中位数分组）')
P('')
P('政策前服务业占比中位数：%.1f%%；高服务业城 %d 个、低服务业城 %d 个'
  % (med, int((pre > med).sum()), int((pre <= med).sum())))
P('')
P('| 结果 Y | did（低服务业城） | did×高服务业 | 交互 p | N |')
P('|---|---|---|---|---|')
for ycol in ['new_jobs_wan', 'gdp_growth', 'income_yuan']:
    r = twfe2(df, ycol, ['did', 'did_x_high'])
    if r:
        b1, s1 = asel(r, 'did')
        b2, s2 = asel(r, 'did_x_high')
        P('| %s | %+.3f (p=%.3f) | %+.3f (p=%.3f) | %.3f | %d |'
          % (ycol, b1, pz(b1, s1), b2, pz(b2, s2), pz(b2, s2), r[2]))
P('')

# ───────── 调节 2：招聘暴露度（W9）─────────
expo = pd.read_csv(EXPO, encoding='utf-8-sig')
expo = expo[expo['n_posts'] >= 10][['city', 'mean_exp', 'sub_share', 'n_posts']]
df = df.merge(expo, on='city', how='left')
sub = df[df['mean_exp'].notna()].copy()
sub['exp_z'] = (sub['mean_exp'] - sub['mean_exp'].mean()) / sub['mean_exp'].std()
sub['did_x_exp'] = sub['did'] * sub['exp_z']

P('## 五、调节效应 2：城市招聘任务暴露度（W9，≥10 岗位的城）')
P('')
P('覆盖城市：%d（处理 %d / 对照 %d）；暴露度已 z 标准化'
  % (sub['city'].nunique(),
     int(sub[sub['treat'] == 1]['city'].nunique()),
     int(sub[sub['treat'] == 0]['city'].nunique())))
P('')
P('| 结果 Y | did | did×暴露度 | 交互 p | N |')
P('|---|---|---|---|---|')
for ycol in ['new_jobs_wan', 'gdp_growth', 'income_yuan']:
    r = twfe2(sub, ycol, ['did', 'did_x_exp'])
    if r:
        b1, s1 = asel(r, 'did')
        b2, s2 = asel(r, 'did_x_exp')
        P('| %s | %+.3f (p=%.3f) | %+.3f (p=%.3f) | %.3f | %d |'
          % (ycol, b1, pz(b1, s1), b2, pz(b2, s2), pz(b2, s2), r[2]))
P('')
P('> **暴露度时点声明**：W9 暴露度来自 2026-10 单时点招聘岗位抽样，属政策后测量，')
P('> 严格说不能作为外生调节变量；本节仅作横截面暗示性证据（suggestive evidence），')
P('> 正文须与 W9 口径声明一并披露，避免过度解读。')
P('')

# ───────── 结论 ─────────
P('## 六、结论')
P('')


def interp(rows, mname, mcol):
    got = [r for r in rows if r['pa'] < 0.10]
    if not got:
        P('- %s 渠道：a 路径（did→M）不显著 → 政策未通过该可观测渠道传导，'
          '如实报告为「该机制不成立」（不强行解释）。' % mname)
        return
    for r in got:
        sig = r['pv'] < 0.10 and ('间接效应显著（Sobel p=%.3f）' % r['pv']) or \
            ('间接效应不显著（Sobel p=%.3f）' % r['pv'])
        P('- %s 渠道（%s）：a=%+.3f 显著，%s；'
          '总效应 %+.3f → 直接效应 %+.3f（衰减 %s）。'
          % (mname, r['y'], r['a'], sig, r['c'], r['cp'],
             '%.0f%%' % (100 * (1 - abs(r['cp']) / abs(r['c']))) if r['c'] != 0 else 'n/a'))
    P('')


interp(rows_A, '产业结构升级', 'tertiary_share')
interp(rows_B, '收入/生产率', 'income_yuan')
P('本报告取代旧版 23 城描述性 r 值，成为正文机制部分的正式依据；')
P('城市级 AI 企业数/产业规模等变量口径混杂，仅保留为附录描述统计。')
P('')
P('产出：`data/analysis/w11_mechanism_report.md`（本文件）')

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L))
print('saved', OUT)
