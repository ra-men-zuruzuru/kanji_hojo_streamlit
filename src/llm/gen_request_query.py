import streamlit as st,os
import json
import ast
from jsonschema import Draft202012Validator
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv

from src.model.model import LLM_Model
from src.util.utils import UserConditions

load_dotenv()

# langsmith llmのフローを見られる
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

ROOT = Path.cwd()


# 店舗取得APIkey
hotpepper_api_key = os.environ.get("HOTPEPPER_API_KEY")
# class State(TypedDict):
#     place: str | None = None
#     genre: str | None = None
#     pop: str | None = None
#     budget: str | None = None
#     condition: str | None = None


# クエリ作成に必要なフラグやパラメータが書かれているjsonを取得
json_schema = None
with open(
    file=Path(ROOT / "src" / "rag" / "gen_request_schema" / "gen_request_schema.json"),
    mode="r",
    encoding="utf-8",
) as f:
    json_schema = json.load(f)

Draft202012Validator.check_schema(json_schema)

# jsonからキーを判別する　説明下手でごめん
area = ["large_area", "middle_area", "small_area"]
query_rule = ["format", "order", "start"]
genre_budget_pop = ["genre", "budget", "party_capacity"]
cond_flag = [
    "wifi",
    "wedding",
    "course",
    "free_drink",
    "free_food",
    "private_room",
    "horigotatsu",
    "tatami",
    "cocktail",
    "shochu",
    "sake",
    "wine",
    "card",
    "non_smoking",
    "charter",
    "ktai",
    "parking",
    "barrier_free",
    "sommelier",
    "night_view",
    "open_air",
    "show",
    "equipment",
    "karaoke",
    "band",
    "tv",
    "lunch",
    "midnight",
    "midnight_meal",
    "english",
    "pet",
    "child",
]


def get_json(docs: str, pgb: UserConditions, user_msg: str):
    # json生成AIにあげるメッセージ
    message = [
        {
            "role": "system",
            "content": "出力は純粋JSONのみ。未知は null、フラグの場合は0。複数指定可能なエリアについてはリストを用いること。でっち上げ禁止。",
        },
        {
            "role": "user",
            "content": f"""
            option:{docs}\n
            area: {pgb.place}\n
            genre: {pgb.genre}\n
            budget: {pgb.budget}\n
            user_message: {user_msg}\n
            The order is always 4.\n
            Output the areas as an array. Multiple selections are allowed.
            """,
        },
    ]

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    resp = client.chat.completions.create(
        temperature=0,
        model=LLM_Model.kimi,
        messages=message,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "name",
                "description": "クエリのjson生成",
                "schema": json_schema,
            },
        },
    )

    params_json = resp.choices[0].message.content
    return params_json


def convert_request(j: dict):
    query = "?"

    for k, v in j.items():
        if k in area:
            query += f"{k}={",".join(v)}"
        elif k in query_rule or k in genre_budget_pop or k in cond_flag:
            query += f"{k}={v}"
        else:
            continue
        query += "&"

    query += "key=" + hotpepper_api_key
    url = "http://webservice.recruit.co.jp/hotpepper/gourmet/v1/" + query

    return url


def main(option_docs: list, pgb: UserConditions, user_msg: str):
    docs = "\n\n".join(option_docs)
    params_json = get_json(docs, pgb, user_msg)

    j_dict = ast.literal_eval(params_json)

    apirquest = convert_request(j_dict)

    return apirquest
