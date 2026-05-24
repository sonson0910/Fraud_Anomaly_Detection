# Final Report Outline - Fraud & Anomaly Detection

## 1. Bối cảnh và mục tiêu

Bài toán 1 của G'Contest yêu cầu phát hiện gian lận và bất thường từ dữ liệu ngân hàng 360 độ: chân dung khách hàng, giao dịch, digital activity và sản phẩm. Dữ liệu thật không có nhãn fraud đã xác minh, nên giải pháp không tự bịa nhãn. Framework được thiết kế như một hệ thống xếp hạng rủi ro để đưa giao dịch vào hàng đợi review.

## 2. Cách tiếp cận nguyên nhân trước model

Thay vì đưa model trước, nhóm xác định ba nhánh nguyên nhân theo đúng assignment:

1. Account takeover / identity compromise: thiết bị mới, IP mới, hoạt động đêm, hoạt động digital ở giai đoạn muộn, người thụ hưởng mới.
2. Unauthorized transfer / capital outflow: chuyển khoản ra ngoài ngân hàng, số tiền lệch baseline, tần suất giao dịch tăng, dòng tiền ra lớn so với CASA.
3. AML network / mule-account pattern: IP/device dùng chung nhiều khách hàng, người thụ hưởng nhận tiền từ nhiều khách hàng, giao dịch số tròn giá trị cao.

`ACTIVITY_NO` được dùng đúng ý nghĩa trong note mentor: số lớn hơn là hành động sau hơn, nên top 10% `ACTIVITY_NO` trong dữ liệu được xem là late-stage digital activity.

## 3. Dữ liệu sử dụng

- Nguồn: `Processed_Data/` và `G_Contest 26'_3rd round assignment.docx`.
- Số giao dịch scored: 1,418,030.
- Số khách hàng scored: 52,488.
- Thời gian dữ liệu: 2019-01-02 đến 2019-12-31.
- Không dùng synthetic data, không dùng synthetic ground truth.

## 4. Framework kỹ thuật

1. Chuẩn hóa schema theo data dictionary, đồng thời xử lý khác biệt tên cột trong file thật như `TRANS_LV1`/`TRXN_LV1` và `LIMIT_AMT_CREDIT`/`LIMIT_AMT`.
2. Tạo baseline hành vi theo từng khách hàng: số tiền trung bình, P95, tần suất ngày, thiết bị/IP/người thụ hưởng đã từng thấy.
3. Liên kết digital activity cùng ngày với giao dịch: activity đêm, late-stage activity, activity liên quan account/authentication.
4. Chấm điểm rule-based theo ba nhánh nguyên nhân.
5. Huấn luyện Isolation Forest không giám sát để bắt giao dịch lệch khỏi phân bố hành vi chung.
6. Điểm cuối = 60% rule score + 40% model anomaly score, sau đó chia band theo capacity review: Low, Medium, High, Critical.
7. xAI engine xuất `top_reasons` và `recommended_action` cho từng giao dịch.

## 5. Kết quả chính

- High/Critical transactions: 35,451.
- Critical transactions: 7,091.
- Khách hàng có High/Critical transaction: 8,662.

Root-cause summary:

- Account takeover / identity compromise: 27,782 High/Critical giao dịch, risk trung bình 44.9/100.
- Unauthorized transfer / capital outflow: 7,099 High/Critical giao dịch, risk trung bình 56.1/100.
- AML network / mule-account pattern: 570 High/Critical giao dịch, risk trung bình 46.8/100.

## 6. Vì sao không báo Precision/Recall như bài có nhãn?

Vì file thật không có confirmed fraud label. Báo precision/recall dựa trên nhãn tự bịa sẽ làm sai bản chất bài thi. Notebook vẫn có phần evaluation, nhưng evaluation ở đây là:

- Schema/data quality checks.
- Coverage của review queue.
- Distribution theo risk band.
- Kiểm tra top-risk có reason codes rõ ràng.
- Chuẩn bị cơ chế nhận feedback từ investigator để hiệu chỉnh threshold/model sau này.

## 7. Gợi ý vận hành thực tế

- Critical: near-real-time hold/manual review, gọi xác minh khách hàng, kiểm tra device/IP/beneficiary.
- High: step-up authentication hoặc manual review trong ngày.
- AML branch: escalation theo mạng lưới IP/device/beneficiary, không nhìn từng giao dịch riêng lẻ.
- Medium: theo dõi tăng cường và nâng cấp nếu lặp lại trong 7 ngày.
- KPI sau khi triển khai: hit rate trong top-K, false positive rate theo phân khúc, review capacity, số case AML escalation, time-to-review.

## 8. Liên hệ với chuẩn nghiệp vụ quốc tế

- FATF Risk-Based Approach for Banking Sector: ngân hàng nên hiểu mức độ rủi ro, ưu tiên nguồn lực vào nơi rủi ro cao và áp dụng biện pháp giảm thiểu tương ứng. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html
- FFIEC Authentication and Access Guidance: với digital banking, kiểm soát nên theo hướng layered security, MFA/step-up authentication và tăng kiểm soát khi giao dịch hoặc truy cập có rủi ro cao. Link: https://www.ffiec.gov/news/press-releases/2021/pr-08-11
- FATF Guidance on Risk-Based Supervision: tránh cách làm tick-box, tập trung vào full spectrum of risks và nơi rủi ro cao hơn. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance-rba-supervision.html

Framework của dự án bám đúng tinh thần này: risk-based, layered controls, review queue theo capacity, và xAI reason codes để investigator kiểm tra được.

## 9. Hạn chế

Đây là framework chuẩn cho dữ liệu không nhãn, không phải model xác nhận fraud. Khi ngân hàng có kết quả review thật, cần đưa label đó quay lại pipeline để hiệu chỉnh threshold, huấn luyện supervised model, và đo Precision@K/Recall@K chính thức.
