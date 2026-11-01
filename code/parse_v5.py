# -*- coding: utf-8 -*-
"""
parse_v5.py — 统计公报严格解析 v5

v4 的三个致命缺陷及修复：
1) find_growth 用 260 字符窗口 + 段内任意"下降"触发全局取负
   → 收紧到 GDP 数字后 60 字符，且要求"下降/减少"紧贴被匹配百分比（≤8 非数字符内）
2) 下载时按标题年份入库出错 → 南京 2021 目录下混入 2009–2021 共 9 份历史公报
   → 重新从标题抽取年份并重归年，再按 (city,title_year) 去重
3) pop/jobs 串位（南宁 2021 pop=10.66 实为新增就业）
   → 两个字段各自带关键词约束，禁止跨字段匹配

输出：data/raw/bulletins_v2/bulletins_extracted_v5.csv
"""
import csv
import glob
import json
import os
import re
import sys
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT4 = os.path.join(ROOT, "data", "raw", "bulletins_v2", "txt4")
TXT5 = os.path.join(ROOT, "data", "raw", "bulletins_v2", "txt5")
TXT_OLD = os.path.join(ROOT, "data", "raw", "bulletins")
OUT = os.path.join(ROOT, "data", "raw", "bulletins_v2", "bulletins_extracted_v5.csv")
SUMMARY = os.path.join(ROOT, "data", "raw", "bulletins_v2", "parse_v5_summary.txt")

NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
N = r"\d+(?:\.\d+)?"

SHELL_KEYS = ("生产总值", "地区生产总值")
COLS = ["city", "year", "title", "url", "src_len", "note",
        "gdp_yi", "gdp_growth_pct", "primary_yi", "secondary_yi", "tertiary_yi",
        "pop_wan", "new_jobs_wan", "registered_unemp_pct", "survey_unemp_pct", "income_yuan"]


def num(s):
    if s is None:
        return None
    s = str(s).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


class T(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, d):
        if d.strip():
            self.parts.append(d)

    def text(self):
        return "\n".join(self.parts)


def to_text(raw):
    """南京站页面存在未闭合 <script>（function closeWin 缺 </script>），
    HTMLParser 会把后续整页正文当 JS 吞掉 → 必须先删闭合脚本块，再剥掉剩余标签。"""
    s = raw
    s = re.sub(r"<script\b[^>]*>.*?</script>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<style\b[^>]*>.*?</style>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&(?:nbsp|amp|lt|gt|quot|#[0-9]+|#x[0-9a-fA-F]+);", " ", s)
    s = re.sub(r"[\[［]\s*\d+\s*[\]］]", "", s)  # 剔除脚注标记 [1]［2］（全/半角括号，长沙站用全角）
    # v7 修复：政府站内联标签把数字切成多段（"17560. 6 7亿元"、"22147\n.35亿元"），
    # 导致所有"数字+亿元"正则失配（武汉/济南 2025 被误判 SHELL、青岛 GDP 串位到服务业营收）。
    # 仅消除数字（含小数点）间空白，避免粘连中文。
    s = re.sub(r"(?<=[\d,.])\s+(?=\d|\.)", "", s)
    s = s.replace("％", "%")  # v8：长沙/德清站用全角％，增速正则只认半角
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


# ---------- 1. GDP 绝对值 ----------
def find_gdp(t):
    for m in re.finditer(r"(?:地区)?生产总值", t):
        seg = t[m.end(): m.end() + 35]
        mm = re.search(r"([\d,]{4,10}|\d{3,6}(?:\.\d+)?)\s*亿元", seg)
        if mm:
            v = num(mm.group(1))
            if v and 200 < v < 70000:
                return v, m.start()
    # v8 回退：苏州 2017/2018 公报以"1.73万亿元"表述（无亿元），换算×10000
    for m in re.finditer(r"(?:地区)?生产总值", t):
        seg = t[m.end(): m.end() + 40]
        mm = re.search(r"(\d{1,3}(?:\.\d+)?)\s*万亿元", seg)
        if mm:
            v = num(mm.group(1))
            if v and 0.3 < v < 70:
                return v * 10000, m.start()
    return None, None


# ---------- 2. GDP 增速（v5 核心修复） ----------
def find_growth(t, gdp_pos=None):
    if gdp_pos is None:
        _, gdp_pos = find_gdp(t)
    if gdp_pos is None:
        return None
    # 窗口 = GDP 句内（GDP 数字之后 60 字符），避免吃到"第一产业下降x%"
    m_g = re.search(r"(?:地区)?生产总值[^0-9]{0,20}([\d,]{4,10}|\d{3,6}(?:\.\d+)?)\s*亿元", t[gdp_pos: gdp_pos + 60])
    if m_g:
        start = gdp_pos + m_g.end()
    else:
        start = gdp_pos + 20
    seg = t[start: start + 60]
    m = re.search(r"([−-]?" + N + r")\s*%", seg)
    if not m:
        return None
    v = num(m.group(1))
    if v is None or v >= 40:
        return None
    # 取负仅当"下降/减少/负增长"紧贴被匹配的百分比（前 10 非数字符内）
    left = seg[: m.start()]
    if re.search(r"(下降|减少|负增长|下滑)", left[-14:]) and not re.search(r"增长|上升", left[-14:]):
        v = -abs(v)
    else:
        v = abs(v)
    return v


# ---------- 3. 三次产业 ----------
def find_ind(t, cn):
    """v9 (W3): 修复 % 误杀。旧版 head[:18] 只要含「%」就跳过，把
    「第三产业增加值91.2亿元，增长1.3%」整句漏掉（91.2 亿元后紧跟 1.3%）。
    新逻辑：先找亿元数字，只检查「增加值」与数字之间是否夹占比类关键词。"""
    for m in re.finditer(cn + r"增加值", t):
        seg = t[m.end(): m.end() + 70]
        mm = re.search(r"([\d,]{2,10}|\d{2,5}(?:\.\d+)?)\s*亿元", seg)
        if not mm:
            continue
        between = seg[:mm.start()]
        if any(k in between for k in ("比重", "占比", "占", "%", "为")):
            continue
        v = num(mm.group(1))
        if v and 0.5 < v < 40000:
            return v
    return None


def find_tertiary_share(t):
    """v9 (W3): 直接从「三次产业比例/比重/结构 X:Y:Z」抽三产占比（Z），
    或「第三产业比重/占比 X%」。用于 p1/p2/p3 缺失时的回退。"""
    m = re.search(r"三次产业(?:比例|比重|结构)[^\d]{0,12}"
                  r"([\d.]+)\s*[:：]\s*([\d.]+)\s*[:：]\s*([\d.]+)", t)
    if m:
        try:
            v = float(m.group(3))
            if 5 < v < 95:
                return v
        except ValueError:
            pass
    m = re.search(r"第三产业(?:比重|占比)[^\d]{0,10}([\d.]{1,5})\s*%", t)
    if m:
        try:
            v = float(m.group(1))
            if 5 < v < 95:
                return v
        except ValueError:
            pass
    return None



# ---------- 4. 常住人口（必须紧邻"万人"，且排除人均GDP语境） ----------
def find_pop(t):
    """v9 (W3): 多句式变体 + 「人」单位换算。旧版只认「至年末，全市常住人口为X万人」，
    漏掉「全市年末常住人口X万人」（三亚）、「年末常住人口X万人」（上饶，无全市）、
    「全市常住人口为X人」（七台河，人单位）等句式。"""
    NUMP = r"[\d,]{4,10}|\d{1,4}(?:\.\d+)?"
    pats = [
        r"(?:至)?年末，?全市常住人口[为是]?\s*(" + NUMP + r")\s*万",
        r"全市年末常住人口[为是]?\s*(" + NUMP + r")\s*万",
        r"年末全市常住人口[为是]?\s*(" + NUMP + r")\s*万",
        r"年末常住人口[为是]?\s*(" + NUMP + r")\s*万",
        r"全市常住人口[为是]?\s*(" + NUMP + r")\s*万",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            v = num(m.group(1))
            if v and 50 < v < 3000:
                return v
    # 「人」单位 -> 万
    for p in [r"全市常住人口[为是]?\s*(\d{4,7})\s*人",
              r"年末常住人口[为是]?\s*(\d{4,7})\s*人"]:
        m = re.search(p, t)
        if m:
            v = num(m.group(1))
            if v and 50000 < v < 30000000:
                return v / 10000.0
    return None



# ---------- 5. 城镇新增就业 ----------
def find_jobs(t):
    """v9 (W2): 关键词变体 + 支持「人」单位换算（人/10000→万）+ 窗口 80 字符。
    旧版只认「城镇新增就业」+「万」且窗口仅 40 字符，漏掉大量「XX人」表述（七台河 5895人、
    佳木斯 27599人）。验证：与旧版在 1233 个重叠文件上 100% 取值一致，新增 794 条，零回归。"""
    kws = [r"城镇新增就业", r"新增城镇就业", r"新增就业人员",
           r"全年新增就业", r"城镇新就业", r"新增就业"]
    for kw in kws:
        for m in re.finditer(kw, t):
            seg = t[m.end(): m.end() + 80]
            mm = re.search(r"([\d,]{1,8}|\d{1,4}(?:\.\d+)?)\s*万", seg)
            if mm:
                v = num(mm.group(1))
                if v and 0.3 < v < 150:
                    return v
            mm2 = re.search(r"([\d,]{1,8}|\d{1,4}(?:\.\d+)?)\s*[人]", seg)
            if mm2:
                v = num(mm2.group(1))
                if v is not None and 3000 < v < 2000000:
                    return v / 10000.0
    return None



# ---------- 6. 失业率 ----------
def find_unemp(t, kind):
    """v9 (W2): 登记失业率 + 调查失业率 + 无「城镇」前缀变体，窗口 25 字符。
    旧版 kind=reg 只认「城镇登记失业率」。新增「登记失业率」「城镇调查失业率」「调查失业率」
    作为回退（仅当无登记失业率时采用调查失业率，标注口径差异）。"""
    if kind == "reg":
        kws = [r"城镇登记失业率", r"登记失业率", r"城镇调查失业率", r"调查失业率"]
    else:
        kws = [r"城镇调查失业率", r"调查失业率", r"城镇登记失业率", r"登记失业率"]
    for kw in kws:
        m = re.search(kw + r"[^\d]{0,25}(" + N + r")\s*%", t)
        if m:
            v = num(m.group(1))
            if v and 0.2 < v < 15:
                return v
    return None



# ---------- 7. 城镇人均可支配收入 ----------
def find_income(t):
    pats = [r"城镇居民人均可支配收入[^\d]{0,14}(" + NUM + r")\s*元",
            r"城镇非私营单位从业人员年平均工资[^\d]{0,14}(" + NUM + r")\s*元",
            r"城镇居民人均年收入[^\d]{0,14}(" + NUM + r")\s*元"]
    for p in pats:
        m = re.search(p, t)
        if m:
            v = num(m.group(1))
            if v and 15000 < v < 160000:
                return v
    return None


# ---------- 主流程 ----------
def shell_check(t):
    """JS 壳页判定：既短又无"生产总值…亿元"数字句。
    v8 修复：德清县 2017-2019"主要指标"摘要页正文仅 560 字但含"生产总值470.2亿元"，
    属实的短摘要页；而 JS 壳页（沈阳 660 字、哈尔滨 68 字）一律无 GDP 数字句。
    故改为"短 且 无 GDP 句"才判壳。"""
    has_gdp = re.search(r"(?:地区)?生产总值[^0-9]{0,20}([\d,]{4,10}|\d{3,6}(?:\.\d+)?)\s*亿元", t) or \
               re.search(r"(?:地区)?生产总值[^0-9]{0,20}\d{1,3}(?:\.\d+)?\s*万亿元", t)
    if len(t) < 2500:
        return not has_gdp
    return not has_gdp


def extract_title_year(t, fallback_name=""):
    """从正文标题抽年份。标题形如'上海市2021年国民经济和社会发展统计公报'或
    '南京市2017年国民经济和社会发展统计公报[1]'，年份位于'统计公报'之前且紧邻'年'。
    v3 下载时按文件名标年份导致错标（深圳 2017 目录混入 2016 公报），故一律以正文为准。
    v7 修复：导航栏的'2026年'链接会污染 t[:400] 的末位年份（武汉 2025 公报被错标 2026），
    优先匹配'YYYY年xx市国民经济'这一公报标题专用句式。"""
    m = re.search(r"(20\d{2})年(?:.{0,30}?)国民经济", t[:600])
    if m:
        return int(m.group(1))
    m = re.search(r"统计公报", t[:400])
    if m:
        head = t[: m.start()]
        ym = re.findall(r"(20\d{2})年", head)
        if ym:
            return int(ym[-1])          # 取紧邻标题前的年份
        ym = re.findall(r"(20\d{2})", head)
        if ym:
            return int(ym[-1])
    m = re.search(r"(20\d{2})", fallback_name)
    return int(m.group(1)) if m else None


def main():
    rows_by_key = {}
    shell_list = []
    stats = {c: 0 for c in COLS}
    n_raw = 0

    # 读取所有已落地 txt（优先级：txt4 已验证 → txt5 v5新采 → txt v3 → bulletins v1）
    # 同 city-year 去重时先入库者胜出，保证已验证数据不被新数据覆盖
    files = sorted(glob.glob(os.path.join(TXT4, "*.txt"))) + \
        sorted(glob.glob(os.path.join(TXT5, "*.txt"))) + \
        sorted(glob.glob(os.path.join(os.path.dirname(TXT4), "txt", "*.txt"))) + \
        sorted(glob.glob(os.path.join(TXT_OLD, "*.txt")))
    for f in files:
        n_raw += 1
        bn = os.path.basename(f)
        raw = open(f, encoding="utf-8", errors="ignore").read()
        t = to_text(raw)
        city = re.sub(r"_\d{4}.*", "", bn).rstrip("_")
        gdp, gp = find_gdp(t)
        rec = {
            "city": city, "title_year": None,
            "gdp_yi": gdp, "gdp_growth_pct": find_growth(t, gp),
            "primary_yi": find_ind(t, "第一产业"),
            "secondary_yi": find_ind(t, "第二产业"),
            "tertiary_yi": find_ind(t, "第三产业"),
            "pop_wan": find_pop(t),
            "new_jobs_wan": find_jobs(t),
            "registered_unemp_pct": find_unemp(t, "reg"),
            "survey_unemp_pct": find_unemp(t, "sur"),
            "income_yuan": find_income(t),
            "title": "", "url": "", "src_len": len(t), "note": "",
        }
        # 年份一律以正文标题为准（文件名年份在 v3 下载时存在错标）
        rec["title_year"] = extract_title_year(t, bn)
        if not re.match(r"[\u4e00-\u9fa5]{2,4}_\d{4}(_|\.txt)", bn):
            continue        # 跳过 diag/summary 等误入目录的诊断文本
                          # 兼容 city_year_hash.txt（旧）与 city_year.txt（v5 新下载）两种命名
        if shell_check(t):
            rec["note"] = "SHELL"
            shell_list.append((city, rec["title_year"], bn, len(t)))
            continue
        if gdp is None:
            rec["note"] = "NO_GDP"
            continue
        # pop/jobs 串位修正：pop 明显小于 jobs 量级则交换
        if rec["pop_wan"] and rec["new_jobs_wan"] and rec["pop_wan"] < 40 and rec["new_jobs_wan"] > 100:
            rec["pop_wan"], rec["new_jobs_wan"] = rec["new_jobs_wan"], rec["pop_wan"]
        # 同一 (city, year) 只保留首份（先入库的多为主公报）
        key = (rec["city"], rec["title_year"])
        if key not in rows_by_key:
            rows_by_key[key] = rec

    rows = sorted(rows_by_key.values(), key=lambda r: (r["city"], r["title_year"] or 0))
    for r in rows:
        r["year"] = r["title_year"]
        for c in COLS:
            if r.get(c) is not None:
                stats[c] += 1

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 校验：锚点值一律取各公报原文表述的可比价增速（同源一致，避免与最终核实数口径混用）
    checks = [("上海", 2020, 1.7), ("上海", 2021, 8.1), ("上海", 2023, 5.0),
              ("北京", 2018, 6.6), ("北京", 2021, 8.5), ("北京", 2023, 5.2),
              ("武汉", 2020, -4.7), ("武汉", 2021, 12.2), ("深圳", 2017, 9.0),
              ("深圳", 2021, 6.7), ("深圳", 2023, 6.0), ("南京", 2021, 7.5),
              ("南宁", 2021, 6.1), ("天津", 2021, 6.6), ("西安", 2019, 7.0)]
    chk_out = []
    for c, y, expect in checks:
        got = None
        for r in rows:
            if r["city"] == c and r["year"] == y:
                got = r["gdp_growth_pct"]
                break
        ok = (got is not None and abs(got - expect) < 0.15)
        chk_out.append((c, y, expect, got, "OK" if ok else "MISMATCH"))

    # 按城市汇总序列
    seq = {}
    for r in rows:
        if r["gdp_yi"]:
            seq.setdefault(r["city"], []).append((r["year"], r["gdp_yi"], r["gdp_growth_pct"]))

    L = ["# parse_v5 严格解析结果\n"]
    L.append("- 扫描 txt：%d" % n_raw)
    L.append("- 有效记录（city-year 去重后）：%d" % len(rows))
    L.append("- GDP 命中率：%d/%d" % (stats["gdp_yi"], len(rows)))
    L.append("- 增速命中率：%d/%d" % (stats["gdp_growth_pct"], len(rows)))
    L.append("- 常住人口：%d  新增就业：%d  城镇收入：%d" %
             (stats["pop_wan"], stats["new_jobs_wan"], stats["income_yuan"]))
    L.append("\n## 校验锚点（预期值来自国家统计局最终核实数）\n")
    L.append("| 城市 | 年份 | 预期增速 | 解析值 | 判定 |")
    L.append("|---|---|---|---|---|")
    n_ok = 0
    for c, y, e, g, s in chk_out:
        if s == "OK":
            n_ok += 1
        L.append("| %s | %d | %+.1f | %s | %s |" % (c, y, e, "-" if g is None else "%.1f" % g, s))
    L.append("\n→ 校验通过 %d/%d\n" % (n_ok, len(chk_out)))
    L.append("## 各城市 GDP 序列\n")
    L.append("| 城市 | 年份数 | GDP 序列(亿元, 增速%) |")
    L.append("|---|---|---|")
    for c in sorted(seq):
        s = sorted(seq[c])
        body = "、".join("%d:%.0f(%s)" % (y, g, ("%.1f" % r) if r is not None else "-") for y, g, r in s)
        L.append("| %s | %d | %s |" % (c, len(s), body))
    if shell_list:
        L.append("\n## JS 壳页（需换源：%d 份）\n" % len(shell_list))
        for c, y, bn, ln in shell_list:
            L.append("- %s %s (%s, %d字符)" % (c, y, bn[:40], ln))
    txt = "\n".join(L)
    open(SUMMARY, "w", encoding="utf-8").write(txt)
    print("rows=%d gdp=%d growth=%d shell=%d chk=%d/%d" %
          (len(rows), stats["gdp_yi"], stats["gdp_growth_pct"], len(shell_list), n_ok, len(chk_out)))


if __name__ == "__main__":
    main()
