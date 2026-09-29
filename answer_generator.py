import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from prompt_builder import build_rag_prompt

# 1. Nạp API Key từ .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[ERROR] Chưa tìm thấy GEMINI_API_KEY trong file .env!")
    print("👉 Hãy mở file .env và điền GEMINI_API_KEY=mã_của_con vào nhé.")
    sys.exit(1)

# Khởi tạo AI Client
client_ai = genai.Client(api_key=api_key)

# 2. Kết nối tới ChromaDB đã lưu
base_dir = Path(__file__).resolve().parent
chroma_db_dir = base_dir / "chroma_db"

chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_collection(
    name="bio10_knowledge", 
    embedding_function=embedding_fn
)

def retrieve_top_k(query: str, k: int = 3) -> list[str]:
    """Truy vấn Top-K đoạn tài liệu liên quan nhất từ Vector DB (IR)"""
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]
    return docs

def answer(query: str) -> str:
    """Hàm suy luận LLM theo đúng lưu đồ thiết kế"""
    # Bước a: Lấy Top-K tài liệu
    context_chunks = retrieve_top_k(query, k=3)
    # IN RA ĐỂ KIỂM TRA XEM NÓ TÌM ĐƯỢC ĐOẠN GÌ
    print("\n🔍 [DEBUG] CÁC ĐOẠN CHROMA TÌM THẤY:")
    for i, c in enumerate(context_chunks):
        print(f"--- Đoạn {i+1} ---:\n{c[:150]}...\n")
    if not context_chunks:
        return "Không tìm thấy dữ liệu liên quan trong tài liệu."

    # Bước b: Dựng Structured Prompt
    structured_prompt = build_rag_prompt(query, context_chunks)

    # Bước c: Gửi tới LLM (kèm cơ chế tự động thử lại nếu server quá tải 503)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client_ai.models.generate_content(
                model="gemini-3.8-flash",
                contents=structured_prompt,
            )
            return response.text
        except Exception as err:
            if "503" in str(err) and attempt < max_retries - 1:
                print(f"[CẢNH BÁO] Máy chủ bận (503), đang thử lại lần {attempt + 1}...")
                time.sleep(2)
                continue
            raise err

if __name__ == "__main__":
    print("\n" + "="*45)
    print("🤖 HỆ THỐNG TRỢ LÝ SINH HỌC 10 (RAG) ĐÃ SẴN SÀNG!")
    print("Gõ câu hỏi của bạn (hoặc gõ 'exit' để thoát)")
    print("="*45 + "\n")

    while True:
        try:
            user_question = input("Học sinh hỏi: ").strip()
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit"]:
                print("Tạm biệt!")
                break

            print("\n⏳ Đang tra cứu tài liệu và suy luận...")
            bot_reply = answer(user_question)
            print(f"\n💡 Trả lời:\n{bot_reply}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")