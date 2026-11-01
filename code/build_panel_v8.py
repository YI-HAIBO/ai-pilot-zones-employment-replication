import os
# -*- coding: utf-8 -*-
"""build_panel_v8.py — 构建 W1 扩样本后的 v8 面板（~295 城 × 2017-2025）。

数据源（按优先级）：
  1. v7 面板（18 处理城 9 年 + 33 已核对照城）——已人工核验，优先级最高
  2. control_v6/ 目录（277 对照城 2020-2025，maptable）
  3. control_hist_full/ + control_hist/ 目录（对照城 2017-2019，tjcn）

处理：
  * 统一解析（parse_v5 正则 + 脚注括号回退 + 增速宽回退）
  * GDP 水平值年际跳变 >1.5× 置空（保留增速）
  * 单元格级补齐（v7 > v6 > hist），冲突保留高优先级
  * 省份/地区（东/中/西/东北）补全
  * 输出完全平衡面板，has_data 标记
输出：data/did_city_panel_2017_2025_v8.csv
"""
import csv, glob, io, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import parse_v5

V7 = os.path.join(ROOT, "data", "did_city_panel_2017_2025_v7.csv")
V8 = os.path.join(ROOT, "data", "did_city_panel_2017_2025_v8.csv")
FRAME = os.path.join(ROOT, "data", "structured", "city_frame_2024.json")
DIR_V6 = os.path.join(ROOT, "data", "raw", "bulletins_v2", "txt5", "control_v6")
DIR_HF = os.path.join(ROOT, "data", "raw", "bulletins_v2", "txt5", "control_hist_full")
DIR_H = os.path.join(ROOT, "data", "raw", "bulletins_v2", "txt5", "control_hist")
EXTRACTED = os.path.join(ROOT, "data", "structured", "control_full_extracted.csv")

YEARS = list(range(2017, 2026))
COLS = ["city", "year", "treat", "batch", "region", "post", "did", "has_data",
        "gdp_yi", "gdp_growth", "pop_wan", "new_jobs_wan", "registered_unemp",
        "primary_yi", "secondary_yi", "tertiary_yi", "income_yuan",
        "budget_yi", "source_url", "tertiary_share", "province"]

PROV_REGION = {
    "北京市": "东部", "天津市": "东部", "河北省": "东部", "上海市": "东部", "江苏省": "东部",
    "浙江省": "东部", "福建省": "东部", "山东省": "东部", "广东省": "东部", "海南省": "东部",
    "山西省": "中部", "安徽省": "中部", "江西省": "中部", "河南省": "中部",
    "湖北省": "中部", "湖南省": "中部",
    "内蒙古自治区": "西部", "广西壮族自治区": "西部", "重庆市": "西部", "四川省": "西部",
    "贵州省": "西部", "云南省": "西部", "西藏自治区": "西部", "陕西省": "西部",
    "甘肃省": "西部", "青海省": "西部", "宁夏回族自治区": "西部", "新疆维吾尔自治区": "西部",
    "辽宁省": "东北", "吉林省": "东北", "黑龙江省": "东北",
}


def fnum(v, nd=2):
    if v is None or isinstance(v, str):
        return ""
    return ("%.{}f".format(nd) % v)


def gstr(v):
    if v is None or isinstance(v, str):
        return ""
    return ("+%.1f" % v) if v >= 0 else ("%.1f" % v)


def parse_one(path):
    """返回 dict 或 None（无 GDP 且无增速）"""
    bn = os.path.basename(path)
    raw = io.open(path, encoding="utf-8", errors="ignore").read()
    t = parse_v5.to_text(raw)
    m = re.match(r"([\u4e00-\u9fa5]{2,6})_(\d{4})_", bn)
    if not m:
        return None
    city, year = m.group(1), m.group(2)
    gdp, gp = parse_v5.find_gdp(t)
    if gdp is None:
        m2 = re.search(r"地区生产总值\[[⑵⒉\d][^\]]{0,200}\][⑵⒉]?([\d,.]{4,12})\s*亿元", t)
        if m2:
            try:
                gdp = float(m2.group(1).replace(",", ""))
            except ValueError:
                pass
    growth = parse_v5.find_growth(t, gp)
    if growth is None:
        m3 = re.search(r"(?:地区)?生产总值[^。]{0,40}?增长\s*([0-9]{1,2}(?:\.[0-9])?)\s*[%％]", t)
        if m3:
            try:
                growth = float(m3.group(1))
            except ValueError:
                pass
    if gdp is None and growth is None:
        return None
    p1 = parse_v5.find_ind(t, "第一产业")
    p2 = parse_v5.find_ind(t, "第二产业")
    p3 = parse_v5.find_ind(t, "第三产业")
    um = re.search(r"URL: (\S+)", raw)
    rec = {"city": city, "year": year, "gdp_yi": gdp, "gdp_growth": growth,
           "pop_wan": parse_v5.find_pop(t),
           "new_jobs_wan": parse_v5.find_jobs(t),
           "registered_unemp": parse_v5.find_unemp(t, "reg"),
           "income_yuan": parse_v5.find_income(t),
           "primary_yi": p1, "secondary_yi": p2, "tertiary_yi": p3,
           "url": um.group(1) if um else ""}
    if None not in (p1, p2, p3) and (p1 + p2 + p3) > 0:
        rec["tertiary_share"] = p3 / (p1 + p2 + p3) * 100.0
    else:
        # W3: p1/p2/p3 缺失时从「三次产业比例 X:Y:Z」直接抽三产占比
        rec["tertiary_share"] = parse_v5.find_tertiary_share(t)
    return rec


def scan(d):
    out = {}
    files = sorted(glob.glob(os.path.join(d, "*.txt")))
    for f in files:
        r = parse_one(f)
        if r is None:
            continue
        key = (r["city"], int(r["year"]))
        if key not in out:
            out[key] = r
    return out, len(files)


def main():
    # ── 1. 省份映射 ──
    city_prov = {}
    if os.path.exists(FRAME):
        j = json.load(io.open(FRAME, encoding="utf-8"))
        for prov, cities in j.get("cities_by_prov", {}).items():
            for c in cities:
                city_prov[c[:-1] if c.endswith("市") else c] = prov
    print("采样框城市-省份映射: %d" % len(city_prov))

    # ── 2. v7 基准 ──
    v7 = list(csv.DictReader(io.open(V7, encoding="utf-8-sig")))
    v7_map = {}
    treat_year = {}
    batch_of = {}
    for r in v7:
        key = (r["city"], int(r["year"]))
        v7_map[key] = r
        if r["treat"] == "1":
            batch_of[r["city"]] = r["batch"]
            if r["post"] == "1":
                y = int(r["year"])
                treat_year[r["city"]] = min(treat_year.get(r["city"], 9999), y)
    print("v7 行数 %d；处理城 %d（批次年份 %s）" % (len(v7), len(treat_year), treat_year))

    # ── 3. 解析新采目录 ──
    rec6, n6 = scan(DIR_V6)
    recH, nH = scan(DIR_HF)
    recH2, nH2 = scan(DIR_H)
    for k, v in recH2.items():      # 旧 control_hist 补缺
        recH.setdefault(k, v)
    print("解析: control_v6 %d 文件 -> %d 记录；历史 %d+%d 文件 -> %d 记录"
          % (n6, len(rec6), nH, nH2, len(recH)))

    allrec = {}
    for src in (recH, rec6):        # 低优先级先写
        for k, v in src.items():
            allrec.setdefault(k, v)

    # ── 4. 城市全集 ──
    cities = set(c for c, y in v7_map) | set(c for c, y in allrec)
    # 处理城固定；对照城剔除处理城同名
    treat_set = set(treat_year)
    cities = {c for c in cities if c not in treat_set} | treat_set
    print("城市全集: %d（处理 %d / 对照 %d）" % (len(cities), len(treat_set), len(cities) - len(treat_set)))

    # ── 5. 单元格级合并 ──
    NUM = ["gdp_yi", "gdp_growth", "pop_wan", "new_jobs_wan", "registered_unemp",
           "primary_yi", "secondary_yi", "tertiary_yi", "income_yuan", "tertiary_share"]
    merged = {}      # (city, year) -> {col: (value, src)}
    for (c, y), r in allrec.items():
        d = {}
        for col in NUM:
            v = r.get(col)
            if v is not None:
                d[col] = (v, "new")
        d["source_url"] = (r.get("url", ""), "new")
        merged[(c, y)] = d
    for (c, y), r in v7_map.items():
        d = merged.setdefault((c, y), {})
        for col in NUM:
            s = (r.get(col) or "").strip()
            if s and s != "NA":
                try:
                    d[col] = (float(s), "v7")
                except ValueError:
                    pass
        su = (r.get("source_url") or "").strip()
        if su:
            d["source_url"] = (su, "v7")

    # ── 6. GDP 水平跳变质量门（仅对本次新采值）──
    by_city = {}
    for (c, y) in merged:
        by_city.setdefault(c, {})[y] = merged[(c, y)]
    n_anom = 0
    for c, d in by_city.items():
        for y in sorted(d):
            cur = d[y].get("gdp_yi")
            if not cur or cur[1] != "new":
                continue
            nb = None
            for dy in (-1, 1):
                v = d.get(y + dy, {}).get("gdp_yi")
                if v:
                    nb = v[0]
                    break
            if nb and (cur[0] > nb * 1.5 or cur[0] < nb / 1.5):
                d[y]["gdp_yi"] = (None, "anomaly")
                n_anom += 1
    print("GDP 水平异常置空: %d" % n_anom)

    # ── 7. 输出平衡面板 ──
    rows = []
    for c in sorted(cities):
        prov = city_prov.get(c, "")
        region = PROV_REGION.get(prov, "")
        is_t = c in treat_set
        batch = batch_of.get(c, "")
        ty = treat_year.get(c)
        for y in YEARS:
            d = merged.get((c, y), {})
            def gv(col):
                v = d.get(col)
                if not v or v[0] is None:
                    return None
                return v[0]
            gdp = gv("gdp_yi")
            gr = gv("gdp_growth")
            has = "1" if (gdp is not None or gr is not None) else "0"
            post = "1" if (is_t and ty is not None and y >= ty) else "0"
            rows.append({
                "city": c, "year": y, "treat": "1" if is_t else "0",
                "batch": batch, "region": region, "post": post,
                "did": "1" if (post == "1") else "0", "has_data": has,
                "gdp_yi": fnum(gdp),
                "gdp_growth": gstr(gr),
                "pop_wan": fnum(gv("pop_wan")),
                "new_jobs_wan": fnum(gv("new_jobs_wan")),
                "registered_unemp": ("%.2f" % gv("registered_unemp")) if gv("registered_unemp") is not None else "",
                "primary_yi": fnum(gv("primary_yi")),
                "secondary_yi": fnum(gv("secondary_yi")),
                "tertiary_yi": fnum(gv("tertiary_yi")),
                "income_yuan": ("%d" % gv("income_yuan")) if gv("income_yuan") is not None else "",
                "budget_yi": "",
                "source_url": (d.get("source_url") or ("", ""))[0],
                "tertiary_share": fnum(gv("tertiary_share"), 1),
                "province": prov,
            })

    with io.open(V8, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    # ── 8. 中间抽取表 ──
    with io.open(EXTRACTED, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "year", "province", "gdp_yi", "gdp_growth", "pop_wan",
                    "new_jobs_wan", "income_yuan", "tertiary_share", "source"])
        for (c, y) in sorted(merged):
            d = merged[(c, y)]
            def s(col):
                v = d.get(col)
                return None if (not v or v[0] is None) else v[0]
            w.writerow([c, y, city_prov.get(c, ""),
                        fnum(s("gdp_yi")), gstr(s("gdp_growth")),
                        fnum(s("pop_wan")), fnum(s("new_jobs_wan")),
                        ("%d" % s("income_yuan")) if s("income_yuan") is not None else "",
                        fnum(s("tertiary_share"), 1),
                        (d.get("source_url") or ("", ""))[0]])

    # ── 9. 统计 ──
    n = len(rows)
    def fill(col):
        return sum(1 for r in rows if (r.get(col) or "").strip() not in ("", "NA"))
    print("\nv8 rows=%d（%d 城 × %d 年）" % (n, len(cities), len(YEARS)))
    for col in ["gdp_yi", "gdp_growth", "pop_wan", "new_jobs_wan", "income_yuan",
                "tertiary_share", "registered_unemp"]:
        k = fill(col)
        print("  %-16s %4d (%.1f%%)" % (col, k, k / n * 100))
    nt = sum(1 for r in rows if r["treat"] == "1")
    print("处理组行 %d / 对照组行 %d；has_data=1 共 %d" % (nt, n - nt, fill("has_data")))

    cov = {}
    for r in rows:
        if r["treat"] == "0":
            cov.setdefault(r["city"], 0)
            if (r["gdp_growth"] or "").strip():
                cov[r["city"]] += 1
    dist = {}
    for c, v in cov.items():
        dist[v] = dist.get(v, 0) + 1
    print("对照城增速年份覆盖分布:", dict(sorted(dist.items())))
    print("saved", V8)


if __name__ == "__main__":
    main()
