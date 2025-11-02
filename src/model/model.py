from dataclasses import dataclass

@dataclass
class LLM_Model:
    llama_instant = "llama-3.1-8b-instant" # レイテンシが速い - 出力の信頼性が低い
    llama_versatile = "llama-3.3-70b-versatile" # instantの信頼性高いバージョン
    openai_gpt120 = "openai/gpt-oss-120b" # この中で一番信頼性高い - ちょっと遅い
    oepnai_gpt20 = "openai/gpt-oss-20b" # 120の廉価版
    groq_compound = "groq/compound" # サーバー側で様々なllmやツールを使って出力 - 最大10個のツールを使うとのことで遅い
    groq_compoundmini = "groq/compound-mini" # compoundの廉価版
    qwen = "qwen/qwen3-32b" 
    kimi = "moonshotai/kimi-k2-instruct-0905" # json出力に強い
