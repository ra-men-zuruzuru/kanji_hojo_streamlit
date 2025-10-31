# -*- coding: utf-8 -*-
import streamlit as st
import time
import pandas as pd
from typing import Optional, List, Dict
from src.llm.llm_normal import stream_graph_updates
from src.util.utils import genres, budgets, UserConditions
from src.db.sql_runner import append_message, get_thread_messages, list_threads

thread_id = st.session_state.get("thread_id")
mode = st.session_state.get("mode")
raw = st.session_state.get("user_conditions")
if isinstance(raw, dict):
    user_cond = UserConditions(**raw)
elif isinstance(raw, UserConditions):
    user_cond = raw
else:
    user_cond = UserConditions()

st.set_option("client.showSidebarNavigation",False)

def write_stream(msg: str):
    for i in msg:
        yield i
        time.sleep(0.005)


def _format_label(t: dict) -> str:
    title = t.get("title") or "無題"
    return f"{title}"


st.set_page_config(page_title="AIチャット", page_icon="🤖", layout="wide")

for m in get_thread_messages(thread_id):
    role = m["role"] if m["role"] in ("user", "assistant") else "assistant"
    with st.chat_message(role):
        st.markdown(m["content"])

with st.sidebar:
    
    if st.button("新しく店舗を探す", use_container_width=True):
        # 必要なら新規スレッド作成のために選択をクリア
        st.session_state.pop("thread_id", None)
        # メイン(app.py)から pages/xx.py へ相対パスで遷移
        st.switch_page("pages/conditions_app.py")

    st.divider()
    st.subheader("スレッド一覧")

    threads: List[Dict] = list_threads()
    
    if not threads:
        st.caption("スレッドはまだありません。条件入力から作成してください。")
        selected_id = None
    else:
        # 現在選択中のスレッドを既定に
        current_id = st.session_state.get("thread_id")
        labels = [_format_label(t) for t in threads]
        ids = [t["id"] for t in threads]
        default_idx = ids.index(current_id) if current_id in ids else 0

        for i, (lbl, tid) in enumerate(zip(labels, ids)):
            if st.button(lbl, key=f"th_{tid}", use_container_width=False):
                if tid != current_id:
                    st.session_state["thread_id"] = tid
                    st.rerun()

# 履歴
if "chat" not in st.session_state:
    st.session_state.chat = []


# 既存履歴の表示
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

data: pd.DataFrame | None = None

if mode == "request":
    user_message = "店舗の条件"
    data = pd.DataFrame(
        {
            "":[
                user_cond.place,
                user_cond.genre,
                user_cond.pop,
                user_cond.budget,
                user_cond.condition,
            ]
        },
        index=["場所", "ジャンル", "参加人数", "予算", "詳細な条件"],
    )
    user_cond.msg = user_cond.msg

elif mode == "chat":
    user_message = st.chat_input("メッセージを入力...")
    user_cond.msg = user_message

if user_message:
    # human
    append_message(thread_id, role="user", content=user_message)

    st.session_state.chat.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)
        if data is not None:
            st.table(data)
        
    # AIにtextを渡す
    ai_message = stream_graph_updates(
        user_cond=user_cond, thread_id=thread_id, mode=mode
    )
    append_message(thread_id, role="assistant", content=ai_message)
    # メッセージを配列に入れる
    st.session_state.chat.append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.write_stream(write_stream(ai_message))

    st.session_state["mode"] = "chat"
    
    st.rerun()
