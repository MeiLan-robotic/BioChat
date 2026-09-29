import os
import sys
from pathlib import Path
from docx import Document


def extract_text_from_element(element) -> str:
    """Trích xuất đệ quy toàn bộ văn bản từ một element trong python-docx,

    bao gồm cả các phần tử văn bản thông thường (w:r) và công thức toán học
    (m:oMath, m:oMathPara).
    """
    text_fragments = []
    for child in element.iter():
        # w:t chứa text thông thường
        if child.tag.endswith("t"):
            if child.text:
                text_fragments.append(child.text)
        # m:t chứa ký hiệu, chỉ số, công thức toán/hóa học
        elif child.tag.endswith("oMath") or child.tag.endswith("oMathPara"):
            math_text = "".join(child.itertext())
            if math_text:
                text_fragments.append(f" {math_text.strip()} ")
    return "".join(text_fragments).strip()


def parse_docx_to_text(input_file: Path, output_file: Path) -> None:
    """Đọc tệp .docx, trích xuất đoạn văn cùng bảng biểu và lưu thành tệp text thuần."""
    if not input_file.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp: {input_file}")

    print(f"[INFO] Bắt đầu xử lý: {input_file.name}")
    doc = Document(input_file)

    extracted_content = []

    # 1. Trích xuất nội dung các đoạn văn (paragraphs)
    for p in doc.paragraphs:
        # Gom text từ cả phần tử XML để tránh bỏ sót các khối equation
        text = extract_text_from_element(p._element)
        if not text:
            text = p.text.strip()
        if text:
            extracted_content.append(text)

    # 2. Trích xuất nội dung từ các bảng biểu (tables) nếu có
    for table in doc.tables:
        for row in table.rows:
            row_data = [
                extract_text_from_element(cell._element) or cell.text.strip()
                for cell in row.cells
            ]
            row_clean = [col for col in row_data if col]
            if row_clean:
                extracted_content.append(" | ".join(row_clean))

    if not extracted_content:
        print("[WARNING] Tài liệu không chứa nội dung văn bản khả dụng.")
        return

    # 3. Ghi kết quả ra tệp output
    full_text = "\n\n".join(extracted_content)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"[SUCCESS] Đã trích xuất {len(extracted_content)} khối nội dung.")
    print(f"[SUCCESS] Tệp kết quả đã lưu tại: {output_file}")


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    if not data_dir.exists():
        print(f"[ERROR] Thư mục dữ liệu không tồn tại: {data_dir}")
        sys.exit(1)

    # Tìm tệp .docx hợp lệ trong thư mục data (bỏ qua tệp tạm mở bởi Word)
    docx_files = [
        f
        for f in data_dir.glob("*.docx")
        if not f.name.startswith("~$") and f.is_file()
    ]

    if not docx_files:
        print(f"[ERROR] Không tìm thấy tệp .docx nào trong: {data_dir}")
        sys.exit(1)

    input_docx = docx_files[0]
    output_txt = data_dir / "sinh10_sample.txt"

    try:
        parse_docx_to_text(input_docx, output_txt)
    except Exception as exc:
        print(f"[ERROR] Xử lý tệp thất bại: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()