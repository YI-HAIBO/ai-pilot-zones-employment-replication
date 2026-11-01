import os
# -*- coding: utf-8 -*-
"""build_geo_w7.py — W7: 从阿里云 DataV 免费市界 GeoJSON 构建城市坐标与空间权重矩阵。

步骤：
  1. 拉取 100000_full.json（省界，含 4 直辖市作为 province 级）
  2. 拉取 31 省 _full.json（含地级市边界）
  3. 计算各市质心（最大多边形包围盒中心）
  4. 输出 city_geo.csv（城市名/adcode/lat/lng）
  5. 构建 k 近邻空间权重矩阵 W（k=5，行标准化），存 W.json
"""
import csv, io, json, math, os, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEO_DIR = os.path.join(ROOT, "data", "geo")
os.makedirs(GEO_DIR, exist_ok=True)
CITY_CSV = os.path.join(GEO_DIR, "city_geo.csv")
W_JSON = os.path.join(GEO_DIR, "W_knn5.json")

UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url, cache):
    if os.path.exists(cache):
        return json.load(io.open(cache, encoding="utf-8"))
    req = urllib.request.Request(url, headers=UA)
    data = urllib.request.urlopen(req, timeout=60).read()
    j = json.loads(data.decode("utf-8"))
    io.open(cache, "w", encoding="utf-8").write(json.dumps(j, ensure_ascii=False))
    return j


def centroid(geom):
    """最大多边形（按包围盒面积）的包围盒中心 -> (lng, lat)"""
    def bbox_area(coords):
        xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
        return (max(xs) - min(xs)) * (max(ys) - min(ys)), (sum(xs) / len(xs), sum(ys) / len(ys))
    polys = []
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"][0]]
    elif geom["type"] == "MultiPolygon":
        polys = [p[0] for p in geom["coordinates"]]
    if not polys:
        return None
    best = max(polys, key=lambda c: bbox_area(c)[0])
    return bbox_area(best)[1]


def main():
    # 1. 全国省界（直辖市作为 province 级出现）
    nat = fetch("https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json",
                os.path.join(GEO_DIR, "nat.json"))
    provs = []
    munis = {}
    for f in nat["features"]:
        p = f["properties"]
        if p.get("level") == "province":
            provs.append((p["adcode"], p["name"]))
            if p["name"] in ("北京市", "天津市", "上海市", "重庆市"):
                c = centroid(f["geometry"])
                if c:
                    munis[p["name"]] = c
    print("省级行政区:", len(provs), "；直辖市中心:", {k: ("%.2f,%.2f" % v) for k, v in munis.items()})

    # 2. 各省城市
    cities = {}   # 短名 -> (adcode, lng, lat, 全名)
    for adcode, pname in provs:
        if pname in munis:
            continue
        try:
            j = fetch("https://geo.datav.aliyun.com/areas_v3/bound/%d_full.json" % adcode,
                      os.path.join(GEO_DIR, "prov_%d.json" % adcode))
        except Exception as e:
            print("  取省 %s 失败: %s" % (pname, e))
            continue
        for f in j["features"]:
            p = f["properties"]
            if p.get("level") != "city":
                continue
            c = centroid(f["geometry"])
            if not c:
                continue
            full = p["name"]
            short = full[:-1] if full.endswith("市") else full
            cities[short] = (p["adcode"], c[0], c[1], full)
    # 直辖市
    for full, c in munis.items():
        short = full[:-1]
        cities[short] = (0, c[0], c[1], full)
    print("城市总数:", len(cities))

    # 3. 输出 city_geo.csv
    with io.open(CITY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["city", "adcode", "lng", "lat", "full_name"])
        for s, (ac, lng, lat, full) in sorted(cities.items()):
            w.writerow([s, ac, "%.4f" % lng, "%.4f" % lat, full])
    print("saved", CITY_CSV)

    # 4. k 近邻权重（k=5，行标准化），经纬度 -> 球面距离 km
    def dist_km(a, b):
        la1, lo1, la2, lo2 = map(math.radians, [a[1], a[0], b[1], b[0]])
        d = 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 +
                         math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
        return 6371 * d

    names = sorted(cities)
    coords = {n: (cities[n][1], cities[n][2]) for n in names}
    W = {}
    for n in names:
        ds = sorted(((dist_km(coords[n], coords[m]), m) for m in names if m != n))
        kn = [m for _, m in ds[:5]]
        W[n] = {m: 1.0 / len(kn) for m in kn}
    io.open(W_JSON, "w", encoding="utf-8").write(json.dumps(W, ensure_ascii=False))
    print("saved", W_JSON, "（%d 城 × k=5 近邻，行标准化）" % len(W))

    # 5. 与面板城市名匹配率
    import pandas as pd
    pan = pd.read_csv(os.path.join(ROOT, "data", "did_city_panel_2017_2025_v8.csv"),
                      encoding="utf-8-sig")
    pcities = set(pan["city"].unique())
    matched = pcities & set(names)
    print("面板城市 %d，地理匹配 %d（%.1f%%）" % (len(pcities), len(matched), len(matched) / len(pcities) * 100))
    miss = sorted(pcities - set(names))
    if miss:
        print("未匹配:", miss[:30])


if __name__ == "__main__":
    main()