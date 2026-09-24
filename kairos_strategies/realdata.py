"""真实 A 股数据接入（自包含，供 kairos-strategies 在真实行情上回测）。

- load_panel: 从本地 CSV 目录读出对齐的价格/成交量面板 -> MarketData（处理停牌/上市日/非正价）。
- fetch_universe: 联网从腾讯公开行情抓取前复权日线到本地（无数据时自动补全）。
- 仅依赖 numpy/pandas + 标准库；解析与抓取均为本项目原创。

数据来自公开行情接口，仅用于研究演示，版权归原作者所有，不构成投资建议。
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import MarketData

# 跨行业流动 A 股池（腾讯/新浪格式：sh/sz + 6 位代码）
UNIVERSE: Dict[str, str] = {
    "sh600519": "贵州茅台", "sz000858": "五粮液", "sh600887": "伊利股份", "sh600809": "山西汾酒",
    "sz002304": "洋河股份", "sh601888": "中国中免", "sh600036": "招商银行", "sh601318": "中国平安",
    "sh601166": "兴业银行", "sh601398": "工商银行", "sh600030": "中信证券", "sz300059": "东方财富",
    "sz000333": "美的集团", "sz000651": "格力电器", "sh600690": "海尔智家", "sh600276": "恒瑞医药",
    "sz300760": "迈瑞医疗", "sh600196": "复星医药", "sz002415": "海康威视", "sz002475": "立讯精密",
    "sh603501": "韦尔股份", "sz300750": "宁德时代", "sz002594": "比亚迪", "sh601012": "隆基绿能",
    "sh600438": "通威股份", "sh601633": "长城汽车", "sh601088": "中国神华", "sh600028": "中国石化",
    "sh601899": "紫金矿业", "sh600585": "海螺水泥", "sh600031": "三一重工", "sh601766": "中国中车",
    "sh600900": "长江电力", "sh600009": "上海机场", "sh601668": "中国建筑", "sh600048": "保利发展",
    "sz002714": "牧原股份", "sz002352": "顺丰控股",
}
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}

# 行业分组（与 kairos-data universe 一致），用于行业中性/均衡回测
SECTOR_GROUPS: Dict[str, List[str]] = {
    "consumer": ["sh600519", "sz000858", "sh600887", "sh600809", "sz002304", "sh601888"],
    "finance": ["sh600036", "sh601318", "sh601166", "sh601398", "sh600030", "sz300059"],
    "appliance": ["sz000333", "sz000651", "sh600690"],
    "pharma": ["sh600276", "sz300760", "sh600196"],
    "tech": ["sz002415", "sz002475", "sh603501", "sz300750"],
    "auto_newenergy": ["sz002594", "sh601012", "sh600438", "sh601633"],
    "energy_material": ["sh601088", "sh600028", "sh601899", "sh600585"],
    "industrial": ["sh600031", "sh601766", "sh600900", "sh600009", "sh601668", "sh600048", "sz002714", "sz002352"],
}
SECTOR_MAP: Dict[str, str] = {s: sec for sec, ms in SECTOR_GROUPS.items() for s in ms}


def load_panel(data_dir: str, drop_incomplete: bool = True,
               periods_per_year: int = 252) -> MarketData:
    """从 CSV 目录加载真实行情为 MarketData（prices=收盘价面板, volumes=成交量面板）。"""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"{data_dir} 下没有 CSV；请先运行 fetch_universe 或 examples/run_real.py")
    prices, volumes = {}, {}
    for f in files:
        sym = f[:-4]
        df = pd.read_csv(os.path.join(data_dir, f), parse_dates=["date"]).set_index("date").sort_index()
        prices[sym] = pd.to_numeric(df["close"], errors="coerce")
        volumes[sym] = pd.to_numeric(df["volume"], errors="coerce")
    p = pd.DataFrame(prices)
    v = pd.DataFrame(volumes)
    p = p.where(p > 0).ffill()              # 非正价(停牌记0)→NaN，用历史前填(无未来函数)
    v = v.fillna(0.0)
    if drop_incomplete:
        start = p.apply(lambda s: s.first_valid_index()).max()
        p, v = p.loc[start:], v.loc[start:]
    return MarketData(prices=p, volumes=v, periods_per_year=periods_per_year, name="ashare_real")


# ------------------------------------------------------------------ 抓取（联网）
def _http_json(url: str, tries: int = 3, timeout: int = 25):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 + k)
    raise RuntimeError(f"抓取失败 {url} ({last})")


def _windows(start: str, end: str, step: int = 700) -> List[Tuple[str, str]]:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out = []
    while s <= e:
        w = min(s + dt.timedelta(days=step), e)
        out.append((s.isoformat(), w.isoformat()))
        s = w + dt.timedelta(days=1)
    return out


def fetch_one(symbol: str, start: str = "2016-01-01", end: Optional[str] = None,
              adjust: str = "qfq") -> pd.DataFrame:
    """腾讯源抓取单只前复权日线（按日期窗口分页拼接）。"""
    end = end or dt.date.today().isoformat()
    rows: Dict[str, tuple] = {}
    for (s, e) in _windows(start, end):
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={symbol},day,{s},{e},800,{adjust}")
        node = (_http_json(url).get("data") or {}).get(symbol) or {}
        for r in (node.get("qfqday") or node.get("day") or []):
            if len(r) >= 6:
                try:
                    rows[r[0]] = (r[0], float(r[1]), float(r[3]), float(r[4]), float(r[2]), float(r[5]))
                except (TypeError, ValueError):
                    pass
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame([rows[k] for k in sorted(rows)],
                      columns=["date", "open", "high", "low", "close", "volume"])
    return df


def fetch_universe(symbols: Sequence[str], out_dir: str, start: str = "2016-01-01",
                   delay: float = 0.3, min_rows: int = 200) -> Dict[str, int]:
    """抓取一篮子并存 CSV 到 out_dir，返回 {symbol: rows}。"""
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for sym in symbols:
        df = fetch_one(sym, start)
        res[sym] = len(df)
        if len(df) >= min_rows:
            df.to_csv(os.path.join(out_dir, f"{sym}.csv"), index=False)
        time.sleep(delay)
    return res
