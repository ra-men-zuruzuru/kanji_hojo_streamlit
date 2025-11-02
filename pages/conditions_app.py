# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
from dataclasses import dataclass, asdict
from typing import Optional, List
from src.util.utils import genres,budgets,UserConditions
from src.db.sql_runner import create_thread

# ============ ページ設定 ============
st.set_page_config(
    page_title="どんな店舗をお探しですか？",
    page_icon="🍽️",
    layout="centered"
)

st.title("どんな店舗をお探しですか？")

# ============ 入力フォーム ============
with st.form("conditions_form", clear_on_submit=False):
    # 宴会会場の場所（自由入力・必須）
    place = st.text_input(
        "宴会会場の場所（必須）",
        placeholder="例：○○駅周辺 / ○○市 など",
        key="place"
    )

    # ジャンル（複数選択・必須）
    genres_sel = st.multiselect(
        "ジャンル（複数選択可・必須）",
        options=genres,
        default=["特になし"],
        help="※「特になし」と他ジャンルを同時に選ぶ場合は「特になし」を外してください",
        key="genres"
    )

    # 参加人数（数値のみ・必須, numericupdown 的）
    pop = st.number_input(
        "参加人数（必須）",
        min_value=1,
        max_value=10000,
        value=4,
        step=1,
        help="半角数字で人数を指定してください",
        key="pop"
    )

    # 予算（プルダウン・必須）
    budget = st.selectbox(
        "予算（必須）",
        options=budgets,
        index=8,  # デフォルト: 「上限なし」
        key="budget"
    )

    # こだわりの条件（任意）
    condition = st.text_area(
        "こだわりの条件（任意）",
        placeholder="例：個室 / 禁煙 / 飲み放題 / プロジェクタあり など",
        height=100,
        key="condition"
    )

    # 画面下部に確定ボタン
    submitted = st.form_submit_button(label="確定",type="primary")

# ============ 確定時の処理（必須チェック） ============
if submitted:
    errors: List[str] = []

    # 場所 必須
    if not (place and place.strip()):
        errors.append("・宴会会場の場所は必須です。")

    # ジャンル 必須（最低1つ）
    if not genres_sel:
        errors.append("・ジャンルは最低1つ選択してください。（「特になし」でも可）")

    # 参加人数 必須（1以上の整数）
    if pop is None or int(pop) < 1:
        errors.append("・参加人数は1以上の整数を入力してください。")

    # 予算 必須（プルダウンから選択）
    if budget not in budgets:
        errors.append("・予算を選択してください。")

    if errors:
        st.error("入力に不備があります。以下をご確認ください：\n" + "\n".join(errors))
    else:
        # 「特になし」＋他ジャンルが同時選択されていたら「特になし」を除外
        if "特になし" in genres_sel and len(genres_sel) > 1:
            genres_sel = [g for g in genres_sel if g != "特になし"]

        # dataclass 仕様に合わせて複数ジャンルは結合して1つの文字列に格納
        joined_genres = ",".join(genres_sel) if genres_sel else None

        thread_id = create_thread(
            f"{place},{joined_genres},{pop}人,{budget},{condition}"
        )
        print(thread_id)
        
        uc = UserConditions(
            place=place.strip() if place else None,
            genre=joined_genres,
            pop=int(pop) if pop is not None else None,
            budget=budget or None,
            condition=condition or None,
            msg=f"開催場所：{place},ジャンル：{joined_genres},参加人数：{pop},予算：{budget},詳細条件：{condition}",
            is_condition_chat=True
        )

        # セッションに保持
        st.session_state["user_conditions"] = uc
        st.session_state["thread_id"] = thread_id
        st.session_state["mode"] = "request"

        # 画面をapp.pyに遷移
        st.switch_page("app.py")

        # st.success("条件を確定しました。下記が UserConditions の内容です。")
        # st.json(asdict(uc))

        # st.caption("※ ジャンルをリストで保持したい場合は、UserConditions.genre を List[str] に変更し、結合処理を外してください。")
else:
    st.info("フォームに入力して「確定」を押してください。")
