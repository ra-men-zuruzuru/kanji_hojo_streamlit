import json
import tiktoken
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader

HERE = Path(__file__).resolve().parent  # /app/src/rag
AREA_DIR = HERE / "hotpepper" / "area"  # /app/src/rag/hotpepper/area
BUDGET_DIR = HERE / "hotpepper" / "budget"
CATEGORY_DIR = HERE / "hotpepper" / "category"
GENRE_DIR = HERE / "hotpepper" / "genre"
OPTION_DIR = HERE / "hotpepper" / "option"

# ragのチャンク分割ルール
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

# トークン化ルール
enc = tiktoken.get_encoding("cl100k_base")  # OpenAI系で広く使う

# ▼埋め込み（日本語OKな多言語モデル）
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# # jsonを読むためのやつ
# json_loader = JSONLoader(
#     file_path="src/rag/shops/shops.json",
#     jq_schema=".shop[]",#shopという配列の中身を読みたいので、こう書く
#     text_content=False,
# )

# # jsonlファイルを一行ずつ解析
# docs = []
# docs_chunked = []

# エリアjsol
area_jsonl_path = []
area_jsonl_path.append(Path(AREA_DIR / "middle_areas.jsonl"))
area_jsonl_path.append(Path(AREA_DIR / "small_areas.jsonl"))
# area_jsonl_path.append(Path("src/rag/hotpepper/area/service_areas.jsonl"))
area_jsonl_path.append(Path(AREA_DIR / "large_areas.jsonl"))
# area_jsonl_path.append(Path("src/rag/hotpepper/area/large_service_areas.jsonl"))

# 予算jsonl
budget_jsonl_path = Path(BUDGET_DIR / "hotpepper_budget_master.jsonl")
# 特集jsonl
special_jsonl_path = Path(CATEGORY_DIR / "hotpepper_special_category_master.jsonl")
# ジャンルjsonl
genre_jsonl_path = Path(GENRE_DIR / "hotpepper_genre_master.jsonl")

# グルメサーチAPIの説明pdf
pdf_path = Path(OPTION_DIR / "hotpepper_gourmet_query_RAG_cheatsheet.pdf")

# エリアストア
vs_doc_area = Chroma(
    embedding_function=emb,
    persist_directory="src/rag/ragdata",  # フォルダに保存
    collection_name="hotpepper_area",
)
# 予算ストア
vs_doc_budget = Chroma(
    embedding_function=emb,
    persist_directory="src/rag/ragdata",  # フォルダに保存
    collection_name="hotpepper_budget",
)
# 特集ストア
vs_doc_special = Chroma(
    embedding_function=emb,
    persist_directory="src/rag/ragdata",  # フォルダに保存
    collection_name="hotpepper_special",
)
# ジャンルストア
vs_doc_genre = Chroma(
    embedding_function=emb,
    persist_directory="src/rag/ragdata",  # フォルダに保存
    collection_name="hotpepper_genre",
)
# getリクエストの作り方ストア
vs_text = Chroma(
    embedding_function=emb,
    persist_directory="src/rag/ragdata",
    collection_name="hotpepper_option",
)

def write_chunking(f_name: str, chunking_docs):
    """トークン化された文字列をファイルに書き出す

    Args:
        f_name (str): _description_
        chunking_docs (_type_): _description_
    """
    if False:
        new_docs = enc.encode(str(chunking_docs))
        with open(f"{f_name}_chunking.txt", "w", encoding="utf-8") as f:
            print(len(new_docs))
            f.write(new_docs)
    return


def _sanitize_metadata(meta: dict | None) -> dict:
    """Chromaに入る形へ正規化。Noneは捨て、list/dictはJSON文字列化。"""
    if not meta:
        return {}
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = str(v)
    return out


def _sanitize_docs(docs: list[Document]) -> list[Document]:
    cleaned = []
    for d in docs:
        cleaned.append(
            Document(
                page_content=d.page_content,
                metadata=_sanitize_metadata(getattr(d, "metadata", None)),
            )
        )
    return cleaned


# チャンク分割
def chunking_json(docs: list):
    """チャンク分割する

    Args:
        docs (list): _description_

    Returns:
        _type_: _description_
    """
    texts = text_splitter.split_documents(docs)
    return texts


# エリアコード追加
def Area():
    docs = []
    for path in area_jsonl_path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)

                # page_content: 検索に使う、人が読んで意味の通るテキスト
                page = (
                    f"小エリア名: {obj.get('small_area_name')} 小エリアコード: {obj.get('small_area_code')} "
                    f"中エリア名: {obj.get('middle_area_name')} 中エリアコード: {obj.get('middle_area_code')} "
                    f"大エリア名: {obj.get('large_area_name')} 大エリアコード: {obj.get('large_area_code')} "
                    # f"サービスエリア名: {obj.get('service_area_name')} サービスエリアコード: {obj.get('service_area_code')} "
                    # f"大サービスエリア名: {obj.get('large_service_area_name')} 大サービスエリアコード: {obj.get('large_service_area_code')}"
                )

                # metadata: 絞り込み用に各コードをそのまま載せる
                meta = {
                    "x": obj.get("small_area_code"),
                    "y": obj.get("middle_area_code"),
                    "z": obj.get("large_area_code"),
                    # "sa": obj.get("service_area_code"),
                    # "ss": obj.get("large_service_area_code"),
                    "name": obj.get("small_area_name"),
                }

                docs.append(Document(page_content=page, metadata=meta))
    docs_chunked = chunking_json(docs)
    docs_chunked = _sanitize_docs(docs_chunked)
    vs_doc_area.add_documents(docs_chunked)
    print("area code complete!")
    return vs_doc_area


# 予算コード
def Budget():
    docs = []
    with open(budget_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # page_content: 検索に使う、人が読んで意味の通るテキスト
            page = f"予算: {obj.get('budget_name')} ({obj.get('budget_code')})"

            # metadata: 絞り込み用に各コードをそのまま載せる
            meta = {
                "budget": obj.get("budget_code"),
            }
            docs.append(Document(page_content=page, metadata=meta))

    docs_chunked = chunking_json(docs)
    docs_chunked = _sanitize_docs(docs_chunked)
    vs_doc_budget.add_documents(docs_chunked)
    print("budget code complete!")
    return vs_doc_budget


# 特集コード
def special():
    docs = []
    with open(special_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            page = (
                f"特集名: {obj.get('special_name')} ({obj.get('special_code')})"
                f"特集カテゴリー名: {obj.get('special_category_name')} ({obj.get('special_category_code')})"
            )

            meta = {
                "special_code": obj.get("special_code"),
                "special_category_code": obj.get("special_category_code"),
            }

            docs.append(Document(page_content=page, metadata=meta))

    docs_chunked = chunking_json(docs)
    docs_chunked = _sanitize_docs(docs_chunked)
    vs_doc_special.add_documents(docs_chunked)
    print("special code complete!")
    return vs_doc_special


# ジャンルコード
def Genre():
    docs = []
    with open(genre_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            page = f"ジャンル名: {obj.get('genre_name')} ({obj.get('genre_code')})"

            meta = {
                "genre_code": obj.get("genre_code"),
            }

            docs.append(Document(page_content=page, metadata=meta))

    docs_chunked = chunking_json(docs)
    docs_chunked = _sanitize_docs(docs_chunked)
    vs_doc_genre.add_documents(docs_chunked)
    print("genre code complete!")
    return vs_doc_genre


# グルメサーチAPIのクエリ生成ルール追加
def Option():
    loader = PyMuPDFLoader(pdf_path)
    text = loader.load_and_split(text_splitter)
    write_chunking("option", text)
    vs_text.add_documents(text)
    print("option complete!")
    return vs_text


# # 西洋料理店を追加
# def Shop():
#     docs = json_loader.load()
#     vs_shop.add_documents(docs)
#     print("shop complete!")
#     return vs_shop

Area()
Budget()
special()
Genre()
Option()
# Shop()

print("ALL Complete!!!")
