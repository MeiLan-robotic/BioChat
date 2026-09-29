from typing import List

def build_rag_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Hàm ghép chỉ dẫn, các đoạn ngữ cảnh Top-K và câu hỏi của học sinh thành một Prompt có cấu trúc.
    """
    joined_context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Bạn là gia sư AI môn Sinh học lớp 10.
Nhiệm vụ của bạn là giải đáp câu hỏi của học sinh CHỈ DỰA VÀO các thông tin có trong phần NGỮ CẢNH dưới đây.

QUY TẮC:
1. Nếu ngữ cảnh không có thông tin để trả lời, hãy nói: "Tài liệu Sinh học hiện tại không đề cập đến nội dung này." Không tự suy đoán hay bịa đặt.
2. Trả lời rõ ràng, chính xác các hiện tượng, phản ứng và thuật ngữ sinh học.

=== NGỮ CẢNH TÀI LIỆU ===
{joined_context}

=== CÂU HỎI CỦA HỌC SINH ===
{query}

=== CÂU TRẢ LỜI ==="""
    return prompt