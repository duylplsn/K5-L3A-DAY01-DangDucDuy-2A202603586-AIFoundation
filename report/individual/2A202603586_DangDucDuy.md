# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603586**
- **Họ và tên:Đặng Đức Duy**
- **SEED (in ở ô đầu notebook):6999**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
| Mô hình | F1 | F1_SUB |
|---|---:|---:|
| 1. BoW + LogReg (tự code) | 0.929 | 0.947 |
| 2. BoW + MLP (deep learning) | 0.950 | 0.974 |
| 3a. Encoder (MiniLM) + kNN | 0.892 | 0.974 |
| 3b. Encoder (MiniLM) + LogReg | 0.957 | 0.947 |
| 3c. Encoder (MiniLM) + MLP head | 0.951 | 0.947 |
| 4a. LLM nhỏ zero-shot | 0.950 | 0.950 |
| 4b. LLM lớn zero-shot | 0.976 | 0.976 |
| 4c. LLM nhỏ few-shot | 0.974 | 0.974 |


## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

1. Tôi dự đoán Logistic Regression có precision cao hơn recall vì dễ bỏ sót spam. Nhưng thực tế precision = 0.867, recall = 1.0. Mô hình nhận diện hết spam trong tập test nhưng lại nhầm một số tin ham thành spam.

2. Tôi dự đoán LLM lớn có F1 cao hơn LLM nhỏ khoảng 0.05–0.1, nhưng thực tế chỉ hơn 0.002 (0.976 so với 0.974). Có thể vì bài toán phân loại SMS khá đơn giản nên model nhỏ đã làm tốt rồi.

3. Tôi đoán BoW đạt khoảng 60–70% accuracy trên tập khó, LLM nhỏ khoảng 75–80%. Thực tế BoW chỉ đạt 54%, còn LLM nhỏ đạt 90%. BoW dễ nhầm khi spam không có từ khóa quen thuộc, còn LLM có thể nhận diện được nhiều cách viết khác nhau hơn.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

BoW tụt nhiều nhất. BoW + LogReg giảm từ F1 = 0.947 trên SUB xuống 0.378 ở tập khó, BoW + MLP giảm từ 0.974 xuống 0.410. Cả hai đều chỉ đạt 54% accuracy. Trong đó BoW + MLP khó nhất với nhóm né từ khóa (0% đúng) và spam viết tự nhiên (10% đúng).

Hai case cụ thể:

- **Case #29:** "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd..." là spam nhưng mô hình đoán sai vì nhiều chữ bị thay bằng số nên không khớp với từ trong vocab.
- **Case #30:** "F R E E e n t r y into our weekly draw..." cũng là spam nhưng từ "free" bị tách thành từng ký tự nên BoW không nhận diện được như bình thường.

Điều này cho thấy BoW phụ thuộc khá nhiều vào cách từ xuất hiện trong dữ liệu, nên dễ sai khi người gửi cố tình thay đổi cách viết.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

Tôi nhận đề #4: Chạy `knn_predict()` với k = 1, 5, 25 và so sánh F1.

Tôi dùng vòng lặp chạy lần lượt 3 giá trị k trên cùng tập test, sau đó tính F1 bằng `f1_score()`.

Kết quả:

- k = 1: F1 = 0.881
- k = 5: F1 = 0.892
- k = 25: F1 = 0.853

k = 5 tốt nhất trong 3 giá trị thử. k quá nhỏ dễ bị ảnh hưởng bởi một tin gần nhất bị nhiễu, còn k quá lớn có thể lấy cả những tin ít liên quan nên làm giảm khả năng phân loại.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

Tôi chọn BoW + MLP vì F1 test = 0.950, F1 SUB = 0.974 và p95 = 0.476 ms/tin, thấp hơn nhiều so với SLA 50 ms. Tốc độ đo được khoảng 27.689 tin/giây, trong khi 10 triệu SMS/ngày tương đương khoảng 116 tin/giây.

Tuy nhiên, mô hình chỉ đạt 54% accuracy trên tập khó. Vì vậy tôi sẽ dùng BoW + MLP để phân loại chính, còn những tin có xác suất không chắc chắn thì chuyển sang LLM kiểm tra thêm.

Cách này tận dụng tốc độ của mô hình nhỏ và khả năng xử lý các tin khó của LLM. Đánh đổi là hệ thống phức tạp hơn, phát sinh chi phí API và cần tách bước LLM khỏi luồng phản hồi chính để tránh vượt SLA.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| TODO 4.1–4.4 | Đọc code phần 4 và hoàn thành các TODO 4.1 đến 4.4. Chỉ sửa phần TODO, giữ nguyên các hàm và cấu hình có sẵn. Ưu tiên code đơn giản, dễ hiểu. | Viết code gọi API, tạo prompt phân loại spam, xử lý nhãn trả về và phân loại hàng loạt. | Yêu cầu kiểm tra lại từng TODO để đảm bảo tận dụng các hàm có sẵn. |
| TODO 4.1 | Template đã có `provider_kwargs()`, tại sao còn truyền `max_tokens` trực tiếp? Hãy sửa để dùng hàm có sẵn. | Sửa `call_openai()` để lấy tham số từ `provider_kwargs()`. | Kiểm tra tính tương thích với các provider và tránh truyền tham số không phù hợp. |
| TODO 4.2–4.3 | Prompt phải trả về đúng `spam` hoặc `ham`. `parse_label()` phải xử lý câu trả lời sai định dạng, chẳng hạn `"Not spam"` phải được hiểu là `ham`. | Chỉnh prompt và bổ sung xử lý phủ định, trả về `None` nếu không xác định được nhãn. | Kiểm tra trường hợp phủ định và câu trả lời không đúng định dạng. |
| TODO 4.4 | Đã có `call_llm()` và `retry_with_backoff()`, hãy sử dụng lại thay vì viết cách gọi API mới. Cần tính thời gian và chi phí cho từng lần gọi. | Sửa hàm phân loại hàng loạt, lưu nhãn dự đoán, thời gian phản hồi, chi phí ước tính và số câu trả lời không hợp lệ. | Kiểm tra việc tái sử dụng hàm có sẵn và các thông tin thống kê. |
| Kiểm tra toàn bộ | Kiểm tra cả 4 TODO theo các test bên dưới. Nếu còn lỗi, chỉ sửa phần TODO, không sửa test. | Rà soát tham số API, prompt, xử lý nhãn và các key trả về của `llm_classify_batch()`. | Dùng các test có sẵn để kiểm tra tính đúng đắn, không thay đổi test. |