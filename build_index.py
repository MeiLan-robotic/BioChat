import sys
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "data" / "sinh10_sample.txt"
    chroma_db_dir = base_dir / "chroma_db"

    if not data_file.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu tại: {data_file}")
        sys.exit(1)

    print(f"[INFO] Bắt đầu đọc dữ liệu từ: {data_file.name}")
    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    if not raw_text.strip():
        print("[ERROR] File text rỗng, không có dữ liệu!")
        sys.exit(1)

    # 1. Cắt nhỏ văn bản (Chunking)
    # chunk_size=600: mỗi đoạn tối đa khoảng 600 ký tự
    # chunk_overlap=100: đoạn sau gối đầu lên đoạn trước 100 ký tự để không bị mất ngữ cảnh
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"[INFO] Đã cắt tài liệu thành {len(chunks)} đoạn nhỏ (chunks).")

    # 2. Khởi tạo kho lưu trữ ChromaDB trên máy
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_dir))

    # Dùng mô hình embedding all-MiniLM-L6-v2 (nhẹ, chạy thẳng trên CPU, không tốn tiền API)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    collection_name = "bio10_knowledge"
    # Xóa collection cũ nếu có để nạp mới tinh
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )

    # 3. Đánh số thứ tự và nạp các đoạn vào ChromaDB
    doc_ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "sinh10.docx", "chunk_id": i} for i in range(len(chunks))]

    print("[INFO] Đang biến chữ thành vector và lưu vào ChromaDB (chờ một lát)...")
    collection.add(
        documents=chunks,
        ids=doc_ids,
        metadatas=metadatas
    )

    print(f"[SUCCESS] Đã lưu thành công {collection.count()} đoạn vector vào ChromaDB!")
    print(f"[SUCCESS] Thư mục kho vector: {chroma_db_dir}")

if __name__ == "__main__":
    main()