import sys
import streamlit as st,os
import time
import datetime
import requests
from pathlib import Path
from src.llm.gen_request_query import main as gen_apirequest
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict, Optional, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import (
    StrOutputParser,
)  # 使うとllmからの応答がcontentだけになる
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import conversation
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.util.utils import UserConditions
from src.model.model import LLM_Model
from src.db.sql_runner import *


load_dotenv()


class State(TypedDict):
    # メッセージのタイプは「リスト」です。アノテーション内の `add_messages`
    # 関数は、この状態キーの更新方法を定義します (この場合、メッセージを
    # 上書きするのではなく、リストに追加します)
    messages: Annotated[list, add_messages]
    # TODO 現状、条件文全て組み合わせてRAG検索機にかけているので、誤った情報を取ってきてしまう。
    # TODO したがってそれぞれの検索機に対応したクエリを挿入する
    place: str | None = None
    genre: str | None = None
    pop: int | None = None
    budget: str | None = None
    condition: str | None = None
    msg: str | None = None
    mode: dict  # "chat" or "request"
    context: str  # RAGの検索結果テキスト
    request_url_base: str | None = (
        None  # key無しのURL（LLM生成）OptionalでNoneも許容する
    )
    request_url_full: str | None = None  # key付きのURL（実呼び出し）
    api_json: dict | None = None  # HotPepperのJSON
    shops: List[dict] | None = None  # results.shop
    top3: List[dict] | None = None  # 上位3件
    sliced_top3: List[dict] | None = None  # 必要なデータだけに絞った店舗情報
    validation_error: str | None = None  # バリデーションエラー
    request_error: str | None = None
    thread_id: str | None = None


# # langsmith llmのフローを見られる
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

datetime_now = datetime.datetime.now()


# tavily llm用WEB検索エンジン
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
search = TavilySearch(max_results=3)

# rag読み込み用
#  埋め込みモデル
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
# 保存済みDBを開く
vs = Chroma(
    persist_directory="src\\rag\\ragdata",
    embedding_function=emb,
    collection_name="hotpepper_api",
)
# エリアコードストア開く
vs_area = Chroma(
    persist_directory="src\\rag\\ragdata",
    embedding_function=emb,
    collection_name="hotpepper_area",
)
# 予算ストア
vs_budget = Chroma(
    persist_directory="src\\rag\\ragdata",
    embedding_function=emb,
    collection_name="hotpepper_budget",
)
# 特集ストア
vs_special = Chroma(
    persist_directory="src\\rag\\ragdata",
    embedding_function=emb,
    collection_name="hotpepper_special",
)
# ジャンルストア
vs_genre = Chroma(
    persist_directory="src\\rag\\ragdata",
    embedding_function=emb,
    collection_name="hotpepper_genre",
)
# クエリ生成ルールストア
vs_option = Chroma(
    persist_directory="src\\rag\\ragdata",
    embedding_function=emb,
    collection_name="hotpepper_option",
)

# # ベクトル検索のretireverの定義
# エリアret (検索機)
retriever_area = vs_area.as_retriever(
    # 検索結果を最大2つ取ってくる
    search_kwargs={
        "k": 2,
    }
)
# 予算ret
retriever_budget = vs_budget.as_retriever(
    search_kwargs={
        "k": 2,
    }
)
# 特集ret
retriever_special = vs_budget.as_retriever(
    search_kwargs={
        "k": 1,
    }
)
# ジャンルret
retriever_genre = vs_genre.as_retriever(
    search_kwargs={
        "k": 3,
    }
)
# クエリ生成ルールret
retriever_option = vs_option.as_retriever(
    search_kwargs={
        "k": 6,
    }
)


# Groqの設定
# 会話用のllm
llm_chat = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model=LLM_Model.openai_gpt120,
    # llmの創造性。高いと創造性が高くなる。低いと論理的になる。
    temperature=0.6,
    # 応答の文字列の長さ制限
    max_tokens=None,
    # 応答を待機する時間
    timeout=None,
    # 要求に失敗したときの最大試行回数
    max_retries=2,
)


# # リクエスト生成用のllm
# llm_gen_request = ChatGroq(
#     api_key=os.environ.get("GROQ_API_KEY"),
#     model=LLM_Model.oepnai_gpt120,
#     # llmの創造性。高いと創造性が高くなる。低いと論理的になる。
#     temperature=0,
#     # 応答の文字列の長さ制限
#     max_tokens=None,
#     # 応答を待機する時間
#     timeout=None,
#     # 要求に失敗したときの最大試行回数
#     max_retries=2,
# )

# 会話履歴を要約するもの
memory = ConversationSummaryBufferMemory(
    llm=llm_chat,
    max_token_limit=300,  # トークン数が制限を超えると要約される
    return_messages=True,  # 会話履歴をリストで返すかどうか
    memory_key="history",  # load_memory_variables() で返るキー名
    input_key="input",  # inputs の中でユーザー入力として拾うキー
    output_key="output",  # outputs の中でモデル出力として拾うキー
)


# ツールに使われる関数
# web検索用
@tool("tavily_search")
def tavily_search_tool(query: str):
    """Web検索を行い、要約済みの結果を返す。引数は自然文の検索クエリ。"""
    return search.invoke(query)


# ツールの定義
tools = [tavily_search_tool]


# チャット時のプロンプト
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"あなたは宴会の幹事の店舗選定をサポートする善良なAI。現在時刻は{datetime_now}。必ず日本語で応答すること。ユーザーの店舗選びの条件が入る。それをもとに別の外部のAIが優れた店をあなたに提示する。\n\n"
            f"modeがrequestの場合のみ：ユーザーの店舗選びの条件が入る。それをもとに別の外部のAIが優れた店をあなたに提示する。あなたは提示された全ての店の選定理由と懸念点を提示すること。もし店が提示されなかったらユーザーに左上の新しく店舗を探すボタンを押して条件を改めるように促す。提示は、店舗名,店舗の特徴,選定理由,の順で。表などを効果的に用いて、簡潔に端的に伝えること。最後に3つの店を比較して一番良い店をまとめること。提示後はユーザに店の詳細を聞いてもいいよなど選択肢を提示する\n\n"
            f"modeがchatの場合：ユーザは提示した店舗に関する質問をする。一般的な会話と変わらない振る舞いをすること。条件の再入力は促さないこと。それ以降はユーザとの会話に場合によって検索ツールを使うこと\n\n"
            f"{tools}\n\nツールです。正しい名前と引数で呼んでね",
        ),
        ("system", "<<<MEMORY>>>\n{summary}\n<<<END>>>"),
        MessagesPlaceholder("history"),  # ここに履歴を丸ごと注入
        (
            "human",
            "モード：{mode}\n\n選定された店：{shops}\nTOP3の店：{top3}\n条件文：{condition}\n\n質問：{input}",
        ),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# tavilyを使えるようにツールを定義(使っていない)
# llm_with_tools = llm.bind_tools(tools)

# エージェント定義 ツールとプロンプトを追加
agent = create_tool_calling_agent(llm_chat, tools, prompt=chat_prompt)
chat_agent = AgentExecutor(tools=tools, agent=agent)

# エージェントなしver
chat_llm = chat_prompt | llm_chat

# リクエストプロンプトapiのurlを生成する
# # TODO 直接クエリをllmに作らせるのは、良くないのでllmにはjsonを作ってもらってそれをpythonでクエリに直すっていう感じで
# request_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             f"あなたの役割：受け取ったドキュメントからHotPepperAPIのGetRequestURLのみ回答。"
#             f"禁止事項：key,keywordクエリは使わないこと。"
#             f"生成ルール：small_areaかmiddle_areaかlarge_areaを用いて出力すること。order=4,format=jsonを指定。ドキュメントにない地名は含めないでください。small_area=X005のように使う。areaとgenreとbudgetと設備・特徴フラグのみ使ってください。"
#             f"ベースurl： http://webservice.recruit.co.jp/hotpepper/gourmet/v1"
#             # f"生成の例1：user:東京スカイツリー、食べ放題、飲み放題　AI:https://webservice.recruit.co.jp/hotpepper/gourmet/v1/?small_area=XA3Y&free_food=1&free_drink=1&order=4 "
#             # f"生成の例2：user:東京スカイツリー、暴飲暴食したい、学生、ワイン　AI:https://webservice.recruit.co.jp/hotpepper/gourmet/v1/?small_area=XA3Y&free_food=1&free_drink=1&wine=1&non_smoking=1&order=4&key=b4c8e78096656406　解説：暴飲暴食したい==食べ放題、飲み放題あり、学生==禁煙席を望む確率が高い。"
#             f"入力された内容から、ドキュメントを逸脱しない範囲で要望に応えられるようにフラグを使うこと。"
#             f"バリデーション人数：認識できない文字列だった場合は'invalid_number_people'と出力すること。"
#             f"バリデーション地名：日本国内に存在しない地名や意味不明な文字列が入力されたら、'invalid_placename'と出力すること。"
#             f"バリデーションはエラー文だけ出力すること。一つのみ出力すること。 まあステップバイステップで考えましょう"
#             f"{tools}\n\nツールです。正しい名前と引数で呼んでね",
#         ),
#         # MessagesPlaceholder("history"),  # ここに履歴を丸ごと注入
#         ("human", "質問：{question}\n\nコンテキスト{context}"),
#     ]
# )

# # llmの設定やプロンプトを代入
# request_chain = request_prompt | llm_chat


def last_user_text(state: State) -> str:
    """最新のユーザー入力を取得"""
    msg = state["messages"][-1]
    return getattr(msg, "content", "") if isinstance(msg, HumanMessage) else str(msg)


# ノードの関数
# def Restaurant_res(state: State):
#     history = memory.load_memory_variables({})["history"]
#     res = chat_chain.invoke(
#         {"input": state["messages"][-1].content, "history": history},
#     )
#     return {"messages": [res]}


def gen_request_node(state: State) -> State:
    """RAGからデータ取得"""
    if state["mode"]["mode"] != "request":
        return state
    ret_text = UserConditions()
    option_docs = []

    # TODO ユーザーの条件の＜場所＞をここで個別に読み込ませる
    q = last_user_text(state)
    docs = retriever_option.invoke(q)
    for d in docs:
        option_docs.append(d.page_content)

    fields = [
        ("place", retriever_area, state["place"]),
        ("genre", retriever_genre, state["genre"]),
        ("budget", retriever_budget, state["budget"]),
    ]

    for attr, retr, user_q in fields:
        docs = retr.invoke(user_q)
        if not docs:
            continue
        value = [d.page_content for d in docs]

        joined_value = "\n".join(value)

        # ret_textクラスの任意の属性にjoined_valueを代入
        setattr(ret_text, attr, joined_value)

    state["request_url_full"] = gen_apirequest(option_docs, ret_text, q)

    return state


def call_hotpepper_api_node(state: State):
    """ホットペッパーAPIから店舗情報をとる。"""
    if state["mode"]["mode"] != "request":
        return state

    # バリデーションエラー検知
    if state.get("validation_error") is not None:
        match state.get("validation_error"):
            case "invalid_number_people":
                print("エラー：正しい人数を入力してください")
            case "invalid_placename":
                print("エラー：国内の開催場所を入力してください")
            case "invalid_time":
                print("エラー：正しい開催日時を入力してください")

        state["shops"] = None
        state["top3"] = None
        return state

    try:
        res = requests.get(state["request_url_full"])
    except Exception as e:
        print("APIリクエストに失敗しました。")
        print(state.get("request_url_full", "url is none"))
        state["request_error"] = "status_not_200"
        return state

    data = res.json()

    try:
        # 店舗情報を全部取る
        state["shops"] = data["results"]["shop"]
        # 上位3位取る[:<num>]スライス
        state["top3"] = state["shops"][:3]
    except Exception as e:
        #
        state["shops"] = None
        # 上位3位取る[:<num>]スライス
        state["top3"] = None
    return state


def slice_shopdata_node(state: State):
    """店舗情報をjsonから辞書に変換"""
    if state["mode"]["mode"] != "request":
        return state

    if state.get("top3") is None:
        return state

    if state.get("sliced_top3") is None:
        state["sliced_top3"] = []

    for item in state["top3"]:
        shop_data_dict = {
            "name": item.get("name"),
            "address": item.get("address"),
            "near_station": item.get("station_name"),
            "genre": item.get("genre", {}).get("name"),
            "genre_catch": item.get("genre", {}).get("catch"),
            "sub_genre": item.get("sub_genre", {}).get("name"),
            "budget": item.get("budget", {}).get("name"),
            "catch": item.get("catch"),
            "capacity": item.get("party_capacity"),
            "access": item.get("access"),
            "booking_url": item.get("urls", {}).get("pc"),
            "other_memo": item.get("other_memo"),
            "shop_detail_memo": item.get("shop_detail_memo"),
            "budget_memo": item.get("budget_memo"),
            "wedding": item.get("wedding"),
            "course": item.get("course"),
            "free_drink": item.get("free_drink"),
            "free_food": item.get("free_food"),
            "private_room": item.get("private_room"),
            "horigotatsu": item.get("horigotatsu"),
            "tatami": item.get("tatami"),
            "card": item.get("card"),
            "non_smoking": item.get("non_smoking"),
            "charter": item.get("charter"),
            "parking": item.get("parking"),
            "barrier_free": item.get("barrier_free"),
            "show": item.get("show"),
            "karaoke": item.get("karaoke"),
            "band": item.get("band"),
            "tv": item.get("tv"),
            "lunch": item.get("lunch"),
            "midnight": item.get("midnight"),
            "english": item.get("english"),
            "pet": item.get("pet"),
            "child": item.get("child"),
            "wifi": item.get("wifi"),
        }
        state["sliced_top3"].append(shop_data_dict)
    # for i in state["sliced_top3"]:
    #     # print(i)

    return state


def response_node(state: State):
    """llmの応答を表示する"""

    history = memory.load_memory_variables({})["history"]

    def to_ai_message(res):
        # res が AIMessage ならそのまま、dict や str の場合は content を取り出して AIMessage にする
        if isinstance(res, AIMessage):
            return res
        if isinstance(res, dict):
            # dict の代表的な出力キーを探す
            content = (
                res.get("content") or res.get("output") or res.get("text") or str(res)
            )
            return AIMessage(content=content)
        return AIMessage(content=str(res))

    if state["mode"]["mode"] == "request":

        try:
            res = chat_llm.invoke(
                {
                    "input": None,
                    "summary": "",
                    "history": history,
                    "condition": last_user_text(state),
                    "shops": None,
                    "top3": state.get("sliced_top3", None),
                    "mode": state.get("mode", {}).get("mode"),
                },
            )
            msg = to_ai_message(res)
            return {"messages": [msg]}

        except Exception as e:
            print("現在、llmにアクセスできません。時間をおいてからお試しください。")
            print(e)

    elif state["mode"]["mode"] == "chat":

        sqlite_summary = get_memory_summary(thread_id=state["thread_id"])
        print(sqlite_summary)
        
        try:
            res = chat_agent.invoke(
                {
                    "input": last_user_text(state),
                    "summary": sqlite_summary,
                    "history": history,
                    "condition": None,
                    "shops": None,
                    "top3": None,
                    "mode": state.get("mode", {}).get("mode"),
                },
            )
            msg = to_ai_message(res)
            return {"messages": [msg]}

        except Exception as e:
            print("現在、llmにアクセスできません。時間をおいてからお試しください。")
            print(e)

    return state


# グラフ構築　フローチャートの枠
workflow = StateGraph(State)

# ノード構築　先に定義した関数をノードとして追加
# workflow.add_node("Restaurant_res", Restaurant_res)
workflow.add_node("gen_request_node", gen_request_node)
workflow.add_node("call_hotpepper_api_node", call_hotpepper_api_node)
workflow.add_node("slice_shopdata_node", slice_shopdata_node)
workflow.add_node("response_node", response_node)

# エッジ構築　ノード間を繋ぐ
workflow.add_edge(START, "gen_request_node")
workflow.add_edge("gen_request_node", "call_hotpepper_api_node")
workflow.add_edge("call_hotpepper_api_node", "slice_shopdata_node")
workflow.add_edge("slice_shopdata_node", "response_node")
workflow.add_edge("response_node", END)

# 実行 MemorySaverがthread_idごとに会話履歴を保存する
app = workflow.compile(checkpointer=MemorySaver())


def memory_delete():
    memory.clear()


# AIに入力を渡し、応答を抜き出して出力
def stream_graph_updates(user_cond: UserConditions, thread_id: str, mode: str) -> str:
    ai_message = ""
    human_message = ""

    for event in app.stream(
        {
            # streamのパラメータは辞書に入れてはいけない
            "messages": [HumanMessage(content=user_cond.msg)],
            "mode": {"mode": mode},
            "place": user_cond.place,
            "genre": user_cond.genre,
            "pop": user_cond.pop,
            "budget": user_cond.budget,
            "condition": user_cond.condition,
            "msg": user_cond.msg,
            "thread_id": thread_id,
        },
        stream_mode="values",
        config={"configurable": {"thread_id": thread_id}},
        # place=user_cond.place,
        # genre=user_cond.genre,
        # pop=user_cond.pop,
        # budget=user_cond.budget,
        # condition=user_cond.condition,
        # msg = user_cond.msg
    ):
        value = event["messages"][-1]

        if isinstance(value, AIMessage):
            ai_message = value.content
        elif isinstance(value, HumanMessage):
            human_message = value.content

    memory.save_context({"input": f"{human_message}"}, {"output": f"{ai_message}"})
    history = memory.load_memory_variables({})["history"]

    update_memory_summary(
        thread_id=thread_id, memory_summary=history[0].content
    )

    return ai_message
