import os
# -*- coding: utf-8 -*-
"""
W9: 招聘数据按「城市 × 任务暴露度」聚合
方法：职业标题的任务型暴露度标注（参照 Felben-Raj-Seamans AI Occupational Exposure
      与 Eloundou et al. GPT-exposure 的思路，因无岗位描述全文，采用三级简化分层）
数据：公共招聘网 v3（22 城、固定关键词设计）+ 智联 SSR commercial（133 城、AI/常规混合关键词）
输出：data/analysis/jobs_city_exposure.csv + data/analysis/w9_jobs_exposure_report.md
"""

import pandas as pd
import numpy as np
import io
import re

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = R + r'\data\analysis\jobs_city_exposure.csv'
OUT_MD = R + r'\data\analysis\w9_jobs_exposure_report.md'

# ───────── 暴露度分层（score = AI 替代暴露度 0-1，越高越易被 AI/LLM 替代）─────────
# 层级设计依据：routine-cognitive（可编码、重复性信息处理）高暴露；
# 面对面/人际/复杂物理任务低暴露；AI 技术岗为互补增强型（低替代暴露）
RULES = [
    # (类, 替代暴露分, 关键词列表) —— 按优先级从前到后匹配
    ('AUG', 0.15, ['算法', '大模型', 'AIGC', '人工智能', 'AI', '机器学习', '深度学习',
                   '计算机视觉', '自然语言', '知识图谱', '语音识别', '语音合成',
                   '数据科学', '数据工程师', '数据仓库', '推荐算法', '风控算法',
                   'OCR', 'RAG', '智能体', 'SLAM', '自动驾驶', '机器人', '无人机',
                   '云计算', '云原生', '量化投资', '标注', '数据分析师', '数据挖掘']),
    ('SUB_H', 0.85, ['数据录入', '录入员', '资料文员', '话务员', '客服', '文员',
                     '前台', '接待员']),
    ('SUB_M', 0.70, ['出纳', '会计助理', '总账会计', '银行柜员', '行政助理', '人事助理',
                     '行政文员', '跟单', '翻译', '文案策划', '薪酬专员', '收银',
                     '财务会计', '会计']),
    ('SUB_L', 0.60, ['电话销售', '网络销售']),
    ('LOW', 0.25, ['销售', '导购', '招聘专员', '人事专员', '客户经理', '采购',
                   '护士', '临床', '医师', '教师', '讲师', '美容', '美发', '厨师',
                   '司机', '保安', '电工', '焊工', '木工', '瓦工', '装修']),
]
CLASS_NAME = {'AUG': 'AI技术互补岗', 'SUB_H': '高替代暴露岗',
              'SUB_M': '中替代暴露岗', 'SUB_L': '偏人际销售岗',
              'LOW': '低暴露岗'}
SUB_CLASSES = {'SUB_H', 'SUB_M', 'SUB_L'}


def classify(title, kw):
    t = str(title or '')
    for cls, score, kws in RULES:
        for k in kws:
            if k in t:
                return cls, score, k
    # 未命中标题规则时回退到抓取关键词
    k2 = str(kw or '')
    for cls, score, kws in RULES:
        for k in kws:
            if k in k2:
                return cls, score, k
    return 'OTHER', 0.30, ''


# ───────── 城市名标准化 ─────────
def norm_city(s):
    s = str(s or '').strip()
    s = s.replace('市', '').replace('县', '')
    drop = ['省', '自治区', '特别行政区']
    for d in drop:
        s = s.replace(d, '')
    if s in ('仙桃', '日喀则', '日本', '东方'):
        return None  # 县级市/噪声/境外，不入面板
    return s or None


# ───────── 载入 ─────────
v3 = pd.read_csv(R + r'\data\raw\jobs\ai_jobs_v3_20261025.csv')
cm = pd.read_csv(R + r'\data\raw\jobs\commercial_jobs_20261025.csv')
panel = pd.read_csv(R + r'\data\did_city_panel_2017_2025_v8.csv', encoding='utf-8-sig')

panel_city_info = (panel.groupby('city')
                   .agg(treat=('treat', 'max'), region=('region', 'first'),
                        province=('province', 'first'))
                   .reset_index())

# v3（公共招聘网，panel_city 已映射，缺失的用 city_name_raw/address 补救）
v3 = v3.copy()
v3['src'] = 'MOHRSS'
v3['sal_lo'] = pd.to_numeric(v3['salary_min'], errors='coerce')
v3['sal_hi'] = pd.to_numeric(v3['salary_max'], errors='coerce')

PANELS = set(panel['city'].unique())

# 区县/县级市 → 母市手工映射（Rule C/D，覆盖 v3 未匹配行的高频区县）
MAN = {
    '官渡区': '昆明', '西山区': '昆明', '盘龙区': '昆明', '五华区': '昆明', '呈贡区': '昆明',
    '路桥区': '台州', '椒江区': '台州', '黄岩区': '台州', '临海市': '台州', '玉环市': '台州',
    '温岭市': '台州', '仙居县': '台州', '三门县': '台州', '天台县': '台州',
    '广陵区': '扬州', '邗江区': '扬州', '江都区': '扬州', '仪征市': '扬州', '高邮市': '扬州', '宝应县': '扬州',
    '宽城区': '长春', '朝阳区': '长春', '南关区': '长春', '绿园区': '长春', '二道区': '长春', '九台区': '长春',
    '吉州区': '吉安', '青原区': '吉安', '于都县': '赣州', '会昌县': '赣州', '永丰县': '吉安',
    '奉新县': '宜春', '万载县': '宜春', '分宜县': '新余',
    '凤阳县': '滁州', '婺源县': '上饶', '德兴市': '上饶',
    '麻城市': '黄冈', '石狮市': '泉州', '晋江市': '泉州', '南安市': '泉州',
    '惠安县': '泉州', '安溪县': '泉州', '德化县': '泉州',
    '翠屏区': '宜宾', '襄城区': '襄阳', '枣阳市': '襄阳', '沭阳县': '宿迁', '宿豫区': '宿迁',
    '海安市': '南通', '海门区': '南通', '崇川区': '南通', '如皋市': '南通', '启东市': '南通', '如东县': '南通',
    '曾都区': '随州', '泰兴市': '泰州', '邳州市': '徐州', '新沂市': '徐州',
    '京口区': '镇江', '句容市': '镇江', '丹棱县': '眉山', '昌宁县': '保山',
    '阳泉市': '阳泉', '荆州市': '荆州', '黄冈市': '黄冈', '大庆市': '大庆',
    '榆次区': '晋中', '利川市': '恩施', '泗阳县': '宿迁', '南丹县': '河池',
}

PROV_SFX = ('省', '市', '州', '地区', '盟', '自治区')


def strip_sfx(s):
    s = str(s or '').strip()
    for suf in PROV_SFX:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def fix_city(row):
    """三级回退：panel_city → city_name_raw 直接命中 → address 文本命中 → 区县手工映射"""
    c = norm_city(row.get('panel_city'))
    if c and c in PANELS:
        return c
    raw = str(row.get('city_name_raw') or '').strip()
    r = strip_sfx(raw)
    if r in PANELS:
        return r
    if raw in PANELS:
        return raw
    a = str(row.get('address') or '')
    for pc in sorted(PANELS, key=len, reverse=True):
        if pc in a:
            return pc
    m = MAN.get(raw)
    if m and m in PANELS:
        return m
    return None


v3['city'] = v3.apply(fix_city, axis=1)
n_fix = int(v3['city'].notna().sum())
print('v3 城市匹配: %d / %d 行' % (n_fix, len(v3)))
v3_missing = v3[v3['city'].isna()]['city_name_raw'].value_counts().head(10)
print('v3 仍未匹配 top10:\n', v3_missing.to_string())

# commercial（智联）
cm = cm.copy()
cm['city'] = cm['city'].apply(norm_city)
cm['src'] = 'Zhaopin'
sal = cm['salary_real'].astype(str).str.extract(r'(\d+)')
cm['sal_lo'] = pd.to_numeric(sal[0], errors='coerce')
cm['sal_hi'] = cm['sal_lo']

# 合并 + 去重（source+job_id 唯一，跨源无重叠已验证）
cols = ['job_id', 'job_name', 'city', 'src', 'keyword', 'sal_lo', 'sal_hi',
        'education', 'publish_time']
jobs = pd.concat([v3[cols], cm[cols]], ignore_index=True)
jobs = jobs.drop_duplicates(subset=['src', 'job_id']).dropna(subset=['city'])
jobs = jobs[jobs['city'] != '']

# 暴露度标注
res = [classify(t, k) for t, k in zip(jobs['job_name'], jobs['keyword'])]
jobs['exp_class'] = [r[0] for r in res]
jobs['exp_score'] = [r[1] for r in res]
jobs['match_kw'] = [r[2] for r in res]
jobs['is_sub'] = jobs['exp_class'].isin(SUB_CLASSES).astype(int)
jobs['is_aug'] = (jobs['exp_class'] == 'AUG').astype(int)

print('== 岗位总量 ==')
print('总条数:', len(jobs), ' 去重后城市数:', jobs['city'].nunique())
print(jobs['exp_class'].value_counts().to_string())
unmatched = jobs[jobs['exp_class'] == 'OTHER']['job_name'].value_counts().head(15)
print('\n未命中规则标题 top15:\n', unmatched.to_string())

# ───────── 城市聚合（全样本）─────────
def agg_city(df, tag):
    g = df.groupby('city').agg(
        n_posts=('job_id', 'count'),
        n_sub=('is_sub', 'sum'),
        n_aug=('is_aug', 'sum'),
        mean_exp=('exp_score', 'mean'),
        sal_mean=('sal_lo', 'mean'),
        sal_p50=('sal_lo', 'median'),
    ).reset_index()
    g['sub_share'] = g['n_sub'] / g['n_posts']
    g['aug_share'] = g['n_aug'] / g['n_posts']
    g['src_tag'] = tag
    return g


g_all = agg_city(jobs, 'ALL')
# v3 子样本（固定关键词设计，跨城可比性最好）
g_v3 = agg_city(jobs[jobs['src'] == 'MOHRSS'], 'MOHRSS')

g = g_all.merge(g_v3[['city', 'n_posts', 'n_sub', 'n_aug', 'sub_share', 'aug_share']],
                on='city', how='left', suffixes=('', '_v3'))
g = g.merge(panel_city_info, on='city', how='left')
g = g.sort_values('n_posts', ascending=False)

# ───────── 报告 ─────────
L = []
P = L.append
P('# W9 招聘岗位任务暴露度聚合报告')
P('')
P('**产出日期**：2026-10-30')
P('**数据源**：公共招聘网（job.mohrss.gov.cn，22 城、固定关键词抽样）'
  ' + 智联招聘 SSR（133 城、AI/常规混合关键词抽样），共 **%d 条**去重岗位' % len(jobs))
P('**方法**：职业标题的任务型暴露度标注。因无岗位描述全文，采用三级简化分层，'
  '思路参照 Felben-Raj-Seamans 的 AI 职业暴露指数（AIOI）与 Eloundou et al. (2023) '
  '的 GPT 职业暴露评分：routine-cognitive（可编码、重复信息处理）任务高替代暴露；'
  '面对面/人际/复杂物理任务低暴露；AI 技术岗为互补增强型（低替代暴露）。')
P('')
P('## 一、口径声明（与 AEL 2025 的 6600 万条严格区分）')
P('')
P('| 维度 | 本文 W9 | AEL 2025 |')
P('|---|---|---|')
P('| 样本量 | %d 条抽样岗位 | 6600 万条全量招聘 |' % len(jobs))
P('| 时间 | 2026-10 单时点横截面 | 多年度面板 |')
P('| 覆盖 | %d 城（面板城市 %d 个） | 全国 |'
  % (jobs['city'].nunique(), g['treat'].notna().sum()))
P('| 抽样设计 | 关键词固定检索（v3：替代岗+AI岗各 ~30 词；智联：AI/常规各 ~8 词），'
  '分页上限内取尽 | 全量 |')
P('| 结论边界 | **只能做暴露度结构与跨城横截面对比，不能推断总量或增速** | 可做弹性/趋势 |')
P('')
P('> **核心限制**：分城岗位数受关键词检索设计与分页上限影响，sub_share/aug_share '
  '的跨城比较仅在 v3 固定关键词子样本内最可比；全样本指标仅作参考。'
  '正文必须明示这是抽样、单时点、小样本证据。')
P('')

P('## 二、暴露度分层定义')
P('')
P('| 层级 | 替代暴露分 | 典型职业 | 含义 |')
P('|---|---|---|---|')
P('| SUB_H 高替代 | 0.85 | 数据录入、客服、文员、前台、话务员 | 纯文本/语音信息处理，LLM 可直接承接 |')
P('| SUB_M 中替代 | 0.70 | 出纳、会计、行政助理、翻译、文案策划 | 结构化信息加工，AI 强辅助 |')
P('| SUB_L 偏人际 | 0.60 | 电话销售、网络销售 | 远程销售，AI 部分替代 |')
P('| LOW 低暴露 | 0.25 | 线下销售、导购、招聘、护理、教学、技工 | 面对面/复杂物理任务 |')
P('| AUG 互补增强 | 0.15 | 算法、大模型、数据科学、机器学习、标注 | AI 创造/增强的岗位 |')
P('')
cnt = jobs['exp_class'].value_counts()
P('标注结果：' + '、'.join('%s %d 条' % (CLASS_NAME.get(k, k), v)
                          for k, v in cnt.items() if k != 'OTHER')
  + ('、未分类 %d 条' % cnt.get('OTHER', 0) if cnt.get('OTHER', 0) else ''))
P('平均替代暴露分：%.3f（替代岗子样本 %.3f）'
  % (jobs['exp_score'].mean(), jobs[jobs['is_sub'] == 1]['exp_score'].mean()))
P('')

P('## 三、城市聚合（覆盖 ≥10 岗位的城市，n=%d）' % int((g['n_posts'] >= 10).sum()))
P('')
top = g[g['n_posts'] >= 10].head(40)
P('| 城市 | 处理 | 岗位 | 替代岗占比 | AI岗占比 | 平均暴露分 | 薪资中位(元) |')
P('|---|:---:|---|---|---|---|---|')
for _, r in top.iterrows():
    P('| %s | %s | %d | %.1f%% | %.1f%% | %.3f | %s |'
      % (r['city'], '处理' if r['treat'] == 1 else ('对照' if pd.notna(r['treat']) else '-'),
         int(r['n_posts']), 100 * r['sub_share'], 100 * r['aug_share'],
         r['mean_exp'], '' if pd.isna(r['sal_p50']) else '%.0f' % r['sal_p50']))
P('')

# 处理 vs 对照 对比（v3 固定关键词子样本内最可比）
P('## 四、处理组 vs 对照组城市（v3 固定关键词子样本）')
P('')
v3c = g[g['n_posts_v3'].notna() & (g['n_posts_v3'] >= 10)]
tv = v3c[v3c['treat'] == 1]
cv = v3c[v3c['treat'] == 0]
if len(tv) and len(cv):
    P('| 指标 | 处理城市(n=%d) | 对照城市(n=%d) | 差异 |' % (len(tv), len(cv)))
    P('|---|---|---|---|')
    for col, nm in [('sub_share_v3', '替代岗占比'), ('aug_share_v3', 'AI岗占比'),
                    ('mean_exp', '平均暴露分'), ('sal_p50', '薪资中位')]:
        a, b = tv[col].mean(), cv[col].mean()
        P('| %s | %.3f | %.3f | %+.3f |' % (nm, a, b, a - b))
    P('')
    P('> 注：v3 检索关键词跨城固定，占比跨城可比性优于全样本；但单时点抽样误差大，'
      '仅供描述与后续 W11 调节变量使用，不作因果解释。')
P('')

# 薪资按暴露层
P('## 五、替代暴露岗 vs AI 技术岗的薪资（智联子样本）')
P('')
zs = jobs[(jobs['src'] == 'Zhaopin') & jobs['sal_lo'].notna() & (jobs['sal_lo'] > 0)]
if len(zs):
    P('| 层级 | 岗位数 | 薪资下限均值(元) | 薪资中位(元) |')
    P('|---|---|---|---|')
    for cls in ['SUB_H', 'SUB_M', 'LOW', 'AUG']:
        sub = zs[zs['exp_class'] == cls]
        if len(sub):
            P('| %s | %d | %.0f | %.0f |' % (CLASS_NAME[cls], len(sub),
                                             sub['sal_lo'].mean(), sub['sal_lo'].median()))
    P('')
    sub_sal = zs[zs['is_sub'] == 1]['sal_lo']
    aug_sal = zs[zs['exp_class'] == 'AUG']['sal_lo']
    if len(sub_sal) and len(aug_sal):
        P('替代暴露岗薪资中位 **%.0f 元** vs AI 技术岗 **%.0f 元**，缺口 %.0f%%。'
          % (sub_sal.median(), aug_sal.median(),
             100 * (aug_sal.median() - sub_sal.median()) / sub_sal.median()))
        P('')
        P('> 若城区政策推高生产率而替代暴露岗薪资未见同步上升，则「红利集中于'
          '技能互补端、风险集中于可替代端」的结构性错配叙事获得微观佐证。')
P('')

P('## 六、产出文件')
P('')
P('- `data/analysis/jobs_city_exposure.csv`：城市 × 暴露度聚合面板（%d 城）' % len(g))
P('- `data/analysis/w9_jobs_exposure_report.md`：本报告')
P('- 暴露度标注明细留存于脚本内 RULES 规则表，可复现')
P('')

g.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
io.open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
print('\nsaved', OUT_CSV, g.shape)
print('saved', OUT_MD)
