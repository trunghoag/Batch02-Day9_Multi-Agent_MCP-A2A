# Bài Tập Thực Hành

Thư mục này chứa các bài tập thực hành cho codelab A2A Multi-Agent.
Trong bản nộp này, hai bài tập chính đã được hoàn thiện để có thể chạy
trực tiếp và đối chiếu với `SOLUTIONS.md`.

## Trạng Thái Hoàn Thành

| Bài tập | File | Trạng thái |
|---|---|---|
| Exercise 2: Tools và Knowledge Base | `exercise_2_tools.py` | Completed |
| Exercise 4: Multi-Agent với Privacy Agent | `exercise_4_multiagent.py` | Completed |

## Exercise 2: Tools và Knowledge Base

Nhiệm vụ đã làm:

1. Thêm entry về luật lao động vào `LEGAL_KNOWLEDGE`.
2. Tạo tool `check_statute_of_limitations`.
3. Kết nối tool vào manual tool-call loop.
4. Test với câu hỏi về thời hiệu khởi kiện.

Chạy:

```bash
uv run python exercises/exercise_2_tools.py
```

## Exercise 4: Multi-Agent với Privacy Agent

Nhiệm vụ đã làm:

1. Implement `privacy_agent`.
2. Thêm conditional routing cho privacy/data/GDPR.
3. Thêm `privacy_agent` vào graph.
4. Aggregate kết quả privacy vào final answer.

Chạy:

```bash
uv run python exercises/exercise_4_multiagent.py
```

## Đáp Án

Đáp án chi tiết nằm trong `SOLUTIONS.md`.

## Yêu Cầu Môi Trường

Để chạy các demo có gọi LLM thật, cần:

```bash
uv sync
cp .env.example .env
```

Sau đó điền `OPENROUTER_API_KEY` thật vào `.env`. Repo không chứa fake key và
không tự sinh fallback answer khi thiếu key.

## Debug

Nếu gặp lỗi:

1. Chạy `uv sync` để cài dependencies.
2. Kiểm tra `.env` có `OPENROUTER_API_KEY` thật.
3. Đọc error message trong terminal.
4. Đối chiếu với `stages/*` và `SOLUTIONS.md`.
