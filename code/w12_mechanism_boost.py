import os
# -*- coding: utf-8 -*-
"""
W12 机制证据链补强：岗位级"替代岗 vs AI 互补岗"薪资双重差分
目的：把 W9 的全样本描述性对比（4001 vs 10001 元）升级为
      控制城市构成 + 学历后的岗位级回归，并检验"错配是否为政策城市特有"。
数据：智联子样本（薪资口径与 W9 §5 一致），岗位级 OLS + 城市聚类。
输出：data/analysis/w12_mechanism_boost_report.md
口径：与 W9 完全一致（RULES 暴露度分层 / classify / norm_city 复制自 w9_jobs_exposure.py）
"""

import pandas as pd
import numpy as np
import io
import statsmodels.formula.api as smf

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_MD = R + r'\data\analysis\w12_mechanism_boost_report.md'

# ── 与 W9 完全一致的暴露度分层 ──
RULES = [
    ('AUG', 0.15, ['算法', '大模型', 'AIGC', '人工智能', 'AI', '机器学习', '深度学习',
                   '计算机视觉', '自然语言', '知识图谱', '语音识别', '语音合成',
                   '数据科学', '数据工程师', '数据仓库', '推荐算法', '风控算法',
                   'OCR', 'RAG', '智能体', 'SLAM', '自动驾驶', '机器人', '无人机',
                   '云计算', '云原生', '量化投资', '标注', '数据分析师', '数据挖掘']),
    ('SUB_H', 0.85, ['数据录入', '录入员', '资料文员', '话务员', '客服', '文员', '前台', '接待员']),
    ('SUB_M', 0.70, ['出纳', '会计助理', '总账会计', '银行柜员', '行政助理', '人事助理',
                     '行政文员', '跟单', '翻译', '文案策划', '薪酬专员', '收银', '财务会计', '会计']),
    ('SUB_L', 0.60, ['电话销售', '网络销售']),
]
SUB_CLASSES = {'SUB_H', 'SUB_M', 'SUB_L'}


def classify(title, kw):
    t = str(title or '')
    for cls, score, kws in RULES:
        for k in kws:
            if k in t:
                return cls, score, k
    k2 = str(kw or '')
    for cls, score, kws in RULES:
        for k in kws:
            if k in k2:
                return cls, score, k
    return 'OTHER', 0.30, ''


def norm_city(s):
    s = str(s or '').strip().replace('市', '').replace('县', '')
    for d in ['省', '自治区', '特别行政区']:
        s = s.replace(d, '')
    if s in ('仙桃', '日喀则', '日本', '东方'):
        return None
    return s or None


# ── 载入 ──
cm = pd.read_csv(R + r'\data\raw\jobs\commercial_jobs_20261025.csv')
panel = pd.read_csv(R + r'\data\did_city_panel_2017_2025_v8.csv', encoding='utf-8-sig')
PANELS = set(panel['city'].unique())
treat_map = panel.groupby('city')['treat'].max().to_dict()

cm['city'] = cm['city'].apply(norm_city)
cm = cm[cm['city'].isin(PANELS)]
sal = cm['salary_real'].astype(str).str.extract(r'(\d+)')
cm['sal_lo'] = pd.to_numeric(sal[0], errors='coerce')
res = [classify(t, k) for t, k in zip(cm['job_name'], cm['keyword'])]
cm['exp_class'] = [r[0] for r in res]
cm['is_sub'] = cm['exp_class'].isin(SUB_CLASSES).astype(int)
cm['is_aug'] = (cm['exp_class'] == 'AUG').astype(int)
cm['treat'] = cm['city'].map(treat_map)
cm['educ'] = cm['education'].astype(str)
cm['lsal'] = np.log(cm['sal_lo'])

# 岗位级样本：替代岗 或 AI 岗，薪资 > 0
d = cm[(cm['sal_lo'] > 0) & ((cm['is_sub'] == 1) | (cm['is_aug'] == 1))].copy()
d['aug_x_treat'] = d['is_aug'] * d['treat']
d['cityf'] = d['city'].astype('category')
d['educf'] = d['educ'].astype('category')

L = []
P = L.append
P('# W12 机制证据链补强：岗位级替代 vs AI 互补薪资双重差分')
P('')
P('**产出日期**：2026-10-31')
P('**目的**：把 W9 的全样本描述性对比（4001 vs 10001 元）升级为控制城市构成 + 学历的')
P('岗位级回归，并检验该薪资错配是否为政策城市所特有（DID 式双重差分）。')
P('**口径**：暴露度分层 / classify / norm_city 与 W9 完全一致；薪资用智联子样本下限。')
P('')
P('## 一、样本与描述')
P('')
P('岗位级样本（替代岗 ∪ AI 岗，薪资>0）：**%d 条，覆盖 %d 个城市**。' % (len(d), d['city'].nunique()))
P('')
P('| 分组 | 替代岗 | AI 互补岗 | 替代岗薪资中位 | AI 岗薪资中位 | 相对比（替代/AI） | 缺口 |')
P('|---|---|---|---|---|---|---|')
for grp, lbl in [(1, '处理城市'), (0, '对照城市')]:
    sub = d[d['treat'] == grp]
    s_ = sub[sub['is_sub'] == 1]['sal_lo']
    a_ = sub[sub['is_aug'] == 1]['sal_lo']
    if len(s_) and len(a_):
        ratio = s_.median() / a_.median()
        gap = 100 * (a_.median() - s_.median()) / s_.median()
        P('| %s | %d 条 | %d 条 | %.0f 元 | %.0f 元 | %.3f | %.0f%% |'
          % (lbl, len(s_), len(a_), s_.median(), a_.median(), ratio, gap))
P('')
P('> 相对比越低表示替代岗相对 AI 岗被压得越低（错配越明显）。')
P('> 处理城 0.455 < 对照城 0.500，方向上支持"政策城市错配更明显"，但需回归检验显著性。')
P('')
P('## 二、岗位级回归（控制城市构成与学历）')
P('')
P('模型：log(薪资) ~ is_aug + is_aug×treat + 城市固定效应 + 学历固定效应，城市聚类稳健 SE。')
P('is_aug 主效应 = AI 互补岗相对替代岗的溢价；is_aug×treat = 该溢价在处理城与对照城之差。')
P('')
P('| 变量 | 系数 | 聚类 SE | t 值 | p 值 |')
P('|---|---|---|---|---|')
mod = smf.ols('lsal ~ is_aug + aug_x_treat + C(cityf) + C(educf)', data=d).fit(
    cov_type='cluster', cov_kwds={'groups': d['city']})
for v in ['is_aug', 'aug_x_treat']:
    P('| %s | %.4f | %.4f | %.2f | %.3f |'
      % ('AI 互补岗（is_aug）' if v == 'is_aug' else 'AI 岗×处理城（交互）',
         mod.params[v], mod.bse[v], mod.tvalues[v], mod.pvalues[v]))
P('| 观测数 | %d | | | |' % int(mod.nobs))
P('| 城市数 | %d | | | |' % d['city'].nunique())
P('')
P('**不含学历控制**（仅城市 FE）：')
P('')
P('| 变量 | 系数 | 聚类 SE | t 值 | p 值 |')
P('|---|---|---|---|---|')
mod2 = smf.ols('lsal ~ is_aug + aug_x_treat + C(cityf)', data=d).fit(
    cov_type='cluster', cov_kwds={'groups': d['city']})
for v in ['is_aug', 'aug_x_treat']:
    P('| %s | %.4f | %.4f | %.2f | %.3f |'
      % ('AI 互补岗（is_aug）' if v == 'is_aug' else 'AI 岗×处理城（交互）',
         mod2.params[v], mod2.bse[v], mod2.tvalues[v], mod2.pvalues[v]))
P('')
P('## 三、结论与诚实边界')
P('')
P('1. **错配存在（稳健）**：is_aug 主效应显著为正（含学历 +0.588，p<0.001；不含学历 +0.699，')
P('   p<0.001），即 AI 互补岗薪资显著高于替代暴露岗（约 +80%），且该缺口在控制城市构成与')
P('   学历后依然显著——不是城市规模或学历构成的伪象。')
P('2. **错配的政策特有性未识别**：is_aug×treat 交互不显著（含学历 p=0.739；不含学历 p=0.678），')
P('   受单时点小样本限制（岗位级仅 499 条、对照城 AI 岗仅 80 条），本文无法从现有数据识别')
P('   "错配是否为政策城市所特有"。描述性相对比（0.455 vs 0.500）的方向支持政策特有，但不显著。')
P('3. **边界声明**：本条证据支持"结构性错配存在"，但不支持"政策引致错配"的因果识别；')
P('   后者需 CLDS 多期个体数据（W10）或更长时点的岗位面板。已同步列入中文稿研究局限。')
P('')
P('> 注：本报告取代 W9 §5 的纯描述性对比作为岗位级机制证据的正式依据；')
P('> W9 §5 的 4001 vs 10001 全样本描述保留为表 11 原始数据。')
P('')

io.open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L[:14]))
print('... saved', OUT_MD)
