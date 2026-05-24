# G'Contest 2026 - Bài toán 1: Fraud & Anomaly Detection

## 1. Chọn nhánh bài toán và mục tiêu

Booklet gợi ý 3 nhánh chính: Fraud & Anomaly Detection, Next Best Financial Offer và Persona-Based Digital Personalization. Nhóm chọn nhánh 1, nhưng thiết kế feature vẫn tận dụng dữ liệu chân dung khách hàng, hành vi số và sản phẩm tài chính để có thể mở rộng sang 2 nhánh còn lại.

Đề án không bắt đầu từ model. Luồng trình bày là: phân tích nguyên nhân có thể gây rủi ro -> gom thành root-cause branches -> xây framework phát hiện -> dùng model và xAI để lượng hóa/giải thích.

## 2. Dữ liệu synthetic

- Khách hàng: 3,000
- Giao dịch: 150,000
- Giao dịch anomaly được cài nhãn kiểm chứng: 2,700
- Tỷ lệ anomaly: 1.80%
- Khoảng thời gian: 2019-01-01 đến 2026-05-31

Dữ liệu gồm 6 nhóm: thông tin khách hàng, giao dịch e-banking, hoạt động số, tiền gửi, tín dụng và thẻ. `CUSTOMER_NUMBER` là key để nối các bảng. `ACTIVITY_NO` được hiểu theo thứ tự hành vi: số nhỏ là bước sớm, số lớn là bước sau/nhạy cảm hơn như add beneficiary hoặc change password. Nhãn chỉ nằm ở file `synthetic_ground_truth.csv`, không trộn vào schema gốc.

## 3. Phân tích nguyên nhân trước khi xây model

Ba nhóm nguyên nhân chính:

1. Account access / identity compromise: thiết bị mới, IP mới, OTP request, add beneficiary, change password.
2. Abnormal transaction behavior: giao dịch ngoài giờ quen thuộc, amount vượt baseline, burst tần suất, dòng tiền ra lớn so với số dư.
3. Network / AML linkage: IP/device dùng bởi nhiều khách hàng, chuyển khoản ngoài hệ thống, giao dịch số tròn lặp lại.

Tóm tắt root-cause trên tập synthetic:

| primary_cause_branch | transaction_count | high_or_critical_count | high_critical_anomaly_count | synthetic_anomaly_count | avg_risk_score | high_critical_precision_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| Network / AML linkage | 56963 | 452 | 452 | 689 | 12.71 | 1.0 |
| Abnormal transaction behavior | 2424 | 314 | 312 | 1059 | 41.59 | 0.9936 |
| Account access / identity compromise | 90462 | 16 | 16 | 952 | 8.06 | 1.0 |
| No strong root cause | 151 | 0 | 0 | 0 | 14.61 | 0.0 |

## 4. Framework phát hiện bất thường

Framework dùng hai lớp:

1. Root-cause rule score: dễ giải thích, bám các nguyên nhân nghiệp vụ ở trên.
2. Isolation Forest: học cấu trúc hành vi tổng thể để bắt các điểm lệch đa chiều.
3. xAI layer: trả `primary_cause_branch`, reason codes, feature importance và khuyến nghị xử lý.

Risk score cuối cùng = 60% model score + 40% rule score.

## 5. Kết quả mô hình

- PR-AUC: 0.9597
- ROC-AUC: 0.9992
- Precision@100: 1.0
- Recall@1000: 0.3681
- Precision tại ngưỡng High/Critical >= 60: 0.9974
- Recall tại ngưỡng High/Critical >= 60: 0.2889

Stability theo thời gian:

| time_period | rows | anomaly_rate | avg_risk_score | high_or_critical_count | precision_at_top_1pct | roc_auc | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-2026 digital acceleration | 48624 | 0.0184 | 10.3341 | 265 | 0.9959 | 0.9992 | 0.963 |
| 2022-2023 recovery | 40317 | 0.019 | 10.3509 | 212 | 0.9826 | 0.9991 | 0.9585 |
| 2020-2021 crisis | 40485 | 0.0182 | 10.3073 | 204 | 0.9802 | 0.9992 | 0.9595 |
| 2019 baseline | 20574 | 0.0145 | 10.6205 | 101 | 0.9756 | 0.9993 | 0.9565 |

## 6. Ví dụ explainability

| transaction_row_id | CUSTOMER_NUMBER | risk_score_0_100 | risk_band | primary_cause_branch | ANOMALY_TYPE | top_reasons |
| --- | --- | --- | --- | --- | --- | --- |
| TRX0117584 | CUS002645 | 94.0 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0107148 | CUS000446 | 93.09 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0103280 | CUS002932 | 91.8 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0046072 | CUS002218 | 91.09 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0120424 | CUS002719 | 89.93 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0028823 | CUS001695 | 89.56 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0094281 | CUS000449 | 88.37 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |
| TRX0125855 | CUS001195 | 88.26 | Critical | Network / AML linkage | MONEY_LAUNDERING_PROXY | Giao dịch ngoài khung giờ thông thường; Số tiền cao bất thường so với lịch sử khách hàng; Dòng tiền ra bất thường so với số dư CASA; Thiết bị mới của khách hàng; Mô hình anomaly đánh giá tổng thể cao |

## 7. Đề xuất triển khai

- Medium risk: theo dõi mềm và so sánh thêm với lịch sử khách hàng.
- High risk: yêu cầu step-up authentication, ưu tiên giao dịch chuyển khoản ngoài hệ thống.
- Critical risk: tạm giữ hoặc đưa vào hàng đợi manual review; nếu có pattern lặp theo IP/device/beneficiary thì chuyển AML escalation.
- Demo/AI assistant: có thể dùng `src/customer_risk_advisor.py` như live demo để nhập `CUSTOMER_NUMBER` hoặc `transaction_row_id`, sau đó trả risk band, nguyên nhân, bằng chứng và hành động đề xuất. Nếu có API LLM, phần này có thể chuyển thành chatbot giải thích cho risk officer.
- KPI nên theo dõi: Precision@K, số case review/ngày, false-positive rate theo phân khúc khách hàng, thời gian xử lý manual review.

## 8. Hạn chế

Kết quả hiện dựa trên synthetic data để minh họa. Khi có dữ liệu thật, cần hiệu chỉnh ngưỡng risk band, contamination rate, rule weight và kiểm định với nhãn fraud/chargeback/manual review thực tế.
