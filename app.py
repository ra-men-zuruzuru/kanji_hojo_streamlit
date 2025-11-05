# -*- coding: utf-8 -*-
import streamlit as st,os
from streamlit_card import card
import time
import pandas as pd
from typing import Optional, List, Dict
from src.llm.llm_normal import stream_graph_updates
from src.util.utils import genres, budgets, UserConditions
from src.db.sql_runner import append_message, get_thread_messages, list_threads

thread_id = st.session_state.get("thread_id")
threads: List[Dict] = list_threads()
if not thread_id and threads:
    thread_id = threads[0]["id"]

mode = st.session_state.get("mode") or 'chat'

raw = st.session_state.get("user_conditions")
if isinstance(raw, dict):
    user_cond = UserConditions(**raw)
elif isinstance(raw, UserConditions):
    user_cond = raw
else:
    user_cond = UserConditions()

st.set_option("client.showSidebarNavigation", False)


def write_stream(msg: str):
    for i in msg:
        yield i
        time.sleep(0.005)


def _format_label(t: dict) -> str:
    title = t.get("title") or "無題"
    return f"{title}"


st.set_page_config(page_title="AIチャット", page_icon="🤖", layout="wide")


if thread_id:
    for m in get_thread_messages(thread_id):
        role = m["role"] if m["role"] in ("user", "assistant") else "assistant"
        with st.chat_message(role):
            st.markdown(m["content"])

# スレッドがなかった時、条件入力ページに誘導
if not threads:
    # ちょいCSSでボタンを大きく・目立たせる
    st.markdown(
        """
        <style>
        .hero { text-align:center; padding: 32px 0 12px; }
        .hero h1 { margin:0; font-size: 2rem; }
        .hero p  { margin:.5rem 0 0; color:#6b7280; } /* slate-500 */
        /* ページ内のボタンを少し大きめに（この画面だけ想定） */
        .stButton>button { padding: 0.9rem 1.25rem; font-size: 1.05rem; border-radius: 12px; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
          <h1>どんなお店を探しますか？</h1>
          <p>場所・ジャンル・人数・予算・こだわりを入力すると、幹事補助チャットボットが最適なお店を提案します。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 中央寄せ配置
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.write("")  # 余白
        go = st.button("🔎 お店をさがす", type="primary", use_container_width=True)
        if go:
            st.switch_page("pages/conditions_app.py")

    st.stop()

else:

    with st.sidebar:
        if st.button("新しく店舗を探す", use_container_width=True):
            # 必要なら新規スレッド作成のために選択をクリア
            st.session_state.pop("thread_id", None)
            # メイン(app.py)から pages/xx.py へ相対パスで遷移
            st.switch_page("pages/conditions_app.py")

        st.divider()
        st.subheader("スレッド一覧")

        # =========スレッド表示===========
        if not threads:
            st.caption("スレッドはまだありません。条件入力から作成してください。")
            selected_id = None
        else:
            # 現在選択中のスレッドを既定に
            current_id = st.session_state.get("thread_id")
            labels = [_format_label(t) for t in threads]
            ids = [t["id"] for t in threads]
            # default_idx = ids.index(current_id) if current_id in ids else 0

            for i, (lbl, tid) in enumerate(zip(labels, ids)):
                if st.button(lbl, key=f"th_{tid}", use_container_width=True,type="secondary"):
                    if tid != current_id:
                        st.session_state["thread_id"] = tid
                        st.rerun()

    # # 履歴
    # if "chat" not in st.session_state:
    #     st.session_state.chat = []

    # # 既存履歴の表示
    # for m in st.session_state.chat:
    #     with st.chat_message(m["role"]):
    #         st.markdown(m["content"])

    user_message = st.chat_input("メッセージを入力...")

    if mode == "request":
        
        
        user_message = f"""
        \n
        店舗の条件\n
        会場の場所　　： {user_cond.place}\n
        ジャンル　　　： {user_cond.genre}\n
        参加人数　　　： {user_cond.pop}\n
        予算　　　　　： {user_cond.budget}\n
        こだわりの条件： {user_cond.condition if user_cond.condition else "なし"}\n
        \n
        """
        user_cond.msg = user_cond.msg

    elif mode == "chat":
        user_cond.msg = user_message

    if user_message:
        # human
        append_message(thread_id, role="user", content=user_message)

        # st.session_state.chat.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        # AIにtextを渡す
        ai_message = stream_graph_updates(
            user_cond=user_cond, thread_id=thread_id, mode=mode
        )
        append_message(thread_id, role="assistant", content=ai_message)
        # メッセージを配列に入れる
        # st.session_state.chat.append({"role": "assistant", "content": ai_message})
        with st.chat_message("assistant"):
            st.write_stream(write_stream(ai_message))

        st.session_state["mode"] = "chat"

        st.rerun()
