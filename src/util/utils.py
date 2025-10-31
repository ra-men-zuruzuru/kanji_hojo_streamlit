from dataclasses import dataclass, asdict
from typing import Optional, List

# ジャンル定数
genres: List[str] = [
    "特になし",
    "居酒屋",
    "ダイニングバー・バル",
    "創作料理",
    "和食",
    "洋食",
    "イタリアン・フレンチ",
    "中華",
    "焼肉・ホルモン",
    "韓国料理",
    "アジア・エスニック料理",
    "各国料理",
    "カラオケ・パーティ",
    "バー・カクテル",
    "ラーメン",
    "お好み焼き・もんじゃ",
    "カフェ・スイーツ",
    "その他グルメ",
]

# 予算定数
budgets: List[str] = [
    "2000円以下",
    "3000円以下",
    "4000円以下",
    "5000円以下",
    "7000円以下",
    "10000円以下",
    "15000円以下",
    "30000円以下",
    "上限なし",
]


@dataclass
class UserConditions:
    place: str | None = None
    genre: str | None = None
    pop: int | None = None
    budget: str | None = None
    condition: str | None = None
    msg: str | None = None
    thread_id: str | None = None
    is_condition_chat: bool | None = True
