# Final Report Outline - Fraud & Anomaly Detection

## 1. Bối cảnh và mục tiêu

Bài toán 1 của G'Contest yêu cầu phát hiện và ngăn chặn gian lận từ dữ liệu ngân hàng 360 độ: chân dung khách hàng, giao dịch, digital activity và sản phẩm. Dữ liệu thật không có nhãn fraud đã xác minh, nên nhóm không tự bịa nhãn confirmed fraud. Thay vào đó, nhóm xây root-cause rules để tạo weak label, sau đó huấn luyện supervised prevention model và đưa ra hành động Allow / Monitor / Step-up / Block.

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
5. Tạo `rule_fraud_label` từ rule score: High/Critical theo nghiệp vụ được xem là fraud weak label; Low rõ ràng được xem là clean weak label; vùng giữa được đánh dấu uncertain.
6. Huấn luyện supervised prevention model học từ weak label này để tự động dự báo xác suất fraud/prevention cho giao dịch mới.
7. Chia band và hành động vận hành: Low = Allow, Medium = Enhanced Monitoring, High = Step-up Authentication, Critical = Block/Hold.
8. xAI engine xuất `top_reasons`, SHAP explanation và `recommended_action` cho từng giao dịch.

Xử lý nghiệp vụ bổ sung:

- `Beneficiary_CUSTOMER_NUMBER` bằng 0/0.0/NaN không được xem là node khách hàng trong mạng lưới money mule. Pipeline tách nhóm này thành non-customer/merchant beneficiary và phân tích tiếp bằng `Merchant_ID_Masked`.
- Các merchant như ví điện tử, QR, telco, utility thường không có customer beneficiary cụ thể; nhóm này không bị coi là missing data mặc định.
- Nếu merchant nội bộ/tín dụng không có customer beneficiary nhưng đi kèm số tiền, giờ hoặc activity bất thường, pipeline đưa vào reason code để kiểm tra thêm.
- Overdue lending/credit được gom thành nhóm rủi ro tín dụng 1-5 theo số ngày quá hạn để bổ sung bối cảnh khách hàng tốt/xấu.

## 5. Kết quả chính

- High/Critical transactions: 179,979.
- Critical transactions: 20,242.
- Khách hàng có High/Critical transaction: 24,888.
- Prevention coverage against rule labels: 90.30%.
- Protected amount by Block/Step-up actions: 3,536,070,412,076.

Root-cause summary:

- Account takeover / identity compromise: 161,830 High/Critical giao dịch, risk trung bình 53.1/100.
- Unauthorized transfer / capital outflow: 16,765 High/Critical giao dịch, risk trung bình 67.6/100.
- AML network / mule-account pattern: 1,384 High/Critical giao dịch, risk trung bình 34.7/100.

Monthly stability backtest:

- Framework được kiểm tra theo từng tháng trong năm 2019.
- `model_metrics.json` có bảng monthly/quarterly stability gồm transaction count, average risk, P95 risk và High/Critical rate.
- Vì dữ liệu chỉ có năm 2019, đây là temporal robustness check, không phải crisis-period validation.

## 6. Vì sao có supervised model khi dữ liệu không có nhãn fraud thật?

Vì file thật không có confirmed fraud label, nhóm không báo rằng weak label là sự thật tuyệt đối. Quy trình đúng là:

1. Dựa trên root-cause analysis để tạo rule score.
2. Từ rule score tạo weak label phục vụ huấn luyện.
3. Supervised model học lại logic nghiệp vụ trên toàn bộ feature set.
4. Dashboard báo coverage/precision theo weak label, không claim đó là confirmed fraud accuracy.

Notebook vẫn có phần evaluation, nhưng evaluation ở đây là:

- Schema/data quality checks.
- Prevention coverage against rule-derived labels.
- Protected amount và số giao dịch được Block/Step-up.
- Confusion matrix, recall, precision và false-positive rate theo weak label.
- Kiểm tra top-risk có reason codes rõ ràng.
- Chuẩn bị cơ chế nhận feedback từ investigator để hiệu chỉnh threshold/model sau này.

Theo định hướng giảm thiểu rủi ro, threshold đang ưu tiên bắt được nhiều giao dịch weak-fraud hơn, tức recall/prevention coverage được ưu tiên trước; false-positive rate vẫn được theo dõi để không làm phiền khách hàng tốt quá mức.

## 7. Gợi ý vận hành thực tế

- Critical: Block/Hold near-real-time, gọi xác minh khách hàng, kiểm tra device/IP/beneficiary.
- High: step-up authentication trước khi cho giao dịch đi tiếp.
- AML branch: escalation theo mạng lưới IP/device/beneficiary, không nhìn từng giao dịch riêng lẻ.
- Medium: theo dõi tăng cường và nâng cấp nếu lặp lại trong 7 ngày.
- KPI sau khi triển khai: prevention coverage, protected amount, hit rate trong top-K, false positive rate theo phân khúc, số case AML escalation, time-to-review.

## 8. Liên hệ với chuẩn nghiệp vụ quốc tế

- FATF Risk-Based Approach for Banking Sector: ngân hàng nên hiểu mức độ rủi ro, ưu tiên nguồn lực vào nơi rủi ro cao và áp dụng biện pháp giảm thiểu tương ứng. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html
- FFIEC Authentication and Access Guidance: với digital banking, kiểm soát nên theo hướng layered security, MFA/step-up authentication và tăng kiểm soát khi giao dịch hoặc truy cập có rủi ro cao. Link: https://www.ffiec.gov/news/press-releases/2021/pr-08-11
- FATF Guidance on Risk-Based Supervision: tránh cách làm tick-box, tập trung vào full spectrum of risks và nơi rủi ro cao hơn. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance-rba-supervision.html

Framework của dự án bám đúng tinh thần này: risk-based, layered controls, review queue theo capacity, và xAI reason codes để investigator kiểm tra được.

## 9. Hạn chế

Đây là framework prevention dùng weak label, không phải model xác nhận fraud tuyệt đối. Khi ngân hàng có kết quả review thật, cần đưa label đó quay lại pipeline để hiệu chỉnh rule, threshold, supervised model và đo Precision@K/Recall@K chính thức.
