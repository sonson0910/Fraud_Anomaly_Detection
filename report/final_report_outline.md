# Final Report Outline - Fraud & Anomaly Detection

## 1. Bối cảnh và mục tiêu

Bài toán 1 của G'Contest yêu cầu phát hiện và ngăn chặn gian lận từ dữ liệu ngân hàng 360 độ: chân dung khách hàng, giao dịch, digital activity và sản phẩm. Dữ liệu thật không có nhãn fraud đã xác minh, nên nhóm không tự bịa nhãn confirmed fraud. Thay vào đó, nhóm xây root-cause rules để tạo weak label, sau đó huấn luyện supervised prevention model và đưa ra hành động Allow / Monitor / Step-up / Block.

## 2. Cách tiếp cận nguyên nhân trước model

Thay vì đưa model trước, nhóm xác định ba nhánh nguyên nhân theo đúng assignment:

1. Account takeover / identity compromise: thiết bị mới, IP mới, hoạt động đêm, hoạt động digital ở giai đoạn muộn, người thụ hưởng mới.
2. Unauthorized transfer / capital outflow: chuyển khoản ra ngoài ngân hàng, số tiền lệch baseline, tần suất giao dịch tăng, dòng tiền ra lớn so với CASA.
3. AML network / mule-account pattern: IP/device dùng chung nhiều khách hàng, người thụ hưởng nhận tiền từ nhiều khách hàng, giao dịch số tròn giá trị cao.

`ACTIVITY_NO` được dùng đúng ý nghĩa trong note mentor: số lớn hơn là hành động sau hơn, nên top 10% `ACTIVITY_NO` trong dữ liệu được xem là late-stage digital activity.

## 3. Quy trình 4 giai đoạn và 11 bước

Giai đoạn 1 - Data Foundation & Feature Engineering:

1. Data Cleaning: đọc dữ liệu thật, chuẩn hóa schema, xử lý beneficiary/merchant, aggregate activity log và ghép product snapshots.
2. Four Baseline Metrics: thiết lập 4 nhóm Transactional, Financial, Environmental, Behavioral.
3. Customer 360 Feature Extraction: rolling window 30/60/90 ngày và một dòng baseline cho mỗi khách hàng.

Giai đoạn 2 - EDA, Thresholding & Auto-Labeling:

4. Baseline EDA: sinh insight/figure từ lift, heatmap, network exposure và Customer 360 để chọn ngưỡng có bằng chứng.
5. IQR Thresholding: dùng `Q3 + 1.5 * IQR` ở cấp khách hàng/ngày/giao dịch.
6. Dynamic Rule Engine: ba nhánh nguyên nhân Account Takeover, Unauthorized Transfer, AML/Mule Network.
7. Risk-Scoring & Auto-Labeling: rule score tạo `rule_fraud_label` weak supervision.

Giai đoạn 3 - Machine Learning & Hybrid Check:

8. ML Training: RandomForest prevention model học từ weak label, theo dõi recall, precision và false-positive rate.
9. Hybrid Matrix: Rule+ML = Block/Hold, Rule-only = Step-up/eKYC, ML-only = Watchlist, No-alert = Allow.

Giai đoạn 4 - Deployment, Dashboard & xAI:

10. Business Dashboard: Streamlit hiển thị prevention impact, protected amount, Customer 360, insight và case review.
11. xAI: reason codes + SHAP surrogate giải thích final hybrid prevention score.

## 4. Dữ liệu sử dụng

- Nguồn: `Processed_Data/` và `G_Contest 26'_3rd round assignment.docx`.
- Số giao dịch scored: 1,418,030.
- Số khách hàng scored: 52,488.
- Thời gian dữ liệu: 2019-01-02 đến 2019-12-31.
- Không dùng synthetic data, không dùng synthetic ground truth.

## 5. Framework kỹ thuật

1. Chuẩn hóa schema theo data dictionary, đồng thời xử lý khác biệt tên cột trong file thật như `TRANS_LV1`/`TRXN_LV1` và `LIMIT_AMT_CREDIT`/`LIMIT_AMT`.
2. Tạo Customer 360 baseline: mỗi khách hàng một dòng với 4 nhóm Transactional, Financial, Environmental, Behavioral.
3. Tính rolling window 30/60/90 ngày để phản ánh thói quen hiện tại thay vì dùng cứng toàn bộ lịch sử.
4. Tạo threshold cá nhân hóa bằng P95, z-score và IQR (`Q3 + 1.5 * IQR`) cho số tiền, tần suất và tổng dòng tiền ngày.
5. Liên kết digital activity cùng ngày với giao dịch: activity đêm, late-stage activity, activity liên quan account/authentication.
6. Chấm điểm rule-based theo ba nhánh nguyên nhân.
7. Tạo `rule_fraud_label` từ rule score: rule score rất cao được xem là fraud weak label; rule score thấp rõ ràng được xem là clean weak label; vùng giữa được đánh dấu uncertain.
8. Huấn luyện supervised prevention model học từ weak label này để tự động dự báo xác suất fraud/prevention cho giao dịch mới.
9. Dùng hybrid Rule + ML matrix để quyết định hành động: Rule+ML = Block/Hold, Rule-only = Step-up/eKYC, ML-only = Special Watchlist, No-alert = Allow.
10. xAI engine xuất `top_reasons`, SHAP explanation và `recommended_action` cho từng giao dịch.

Xử lý nghiệp vụ bổ sung:

- `Beneficiary_CUSTOMER_NUMBER` bằng 0/0.0/NaN không được xem là node khách hàng trong mạng lưới money mule. Pipeline tách nhóm này thành non-customer/merchant beneficiary và phân tích tiếp bằng `Merchant_ID_Masked`.
- Các merchant như ví điện tử, QR, telco, utility thường không có customer beneficiary cụ thể; nhóm này không bị coi là missing data mặc định.
- Nếu merchant nội bộ/tín dụng không có customer beneficiary nhưng đi kèm số tiền, giờ hoặc activity bất thường, pipeline đưa vào reason code để kiểm tra thêm.
- Overdue lending/credit được gom thành nhóm rủi ro tín dụng 1-5 theo số ngày quá hạn để bổ sung bối cảnh khách hàng tốt/xấu.
- `outputs/customer_360_baseline.csv` là master data để giải thích baseline 360 độ và làm nền cho dashboard/report.
- `outputs/figures/hybrid_decision_matrix.png`, `rolling_window_baseline.png` và `customer_360_credit_risk_group.png` minh họa rõ phần vận hành, rolling baseline và bối cảnh tín dụng.

## 6. Kết quả chính

- High/Critical transactions: 109,155.
- Critical transactions: 79,772.
- Khách hàng có High/Critical transaction: 27,864.
- Prevention coverage against rule labels: 100.00%.
- Protected amount by Block/Step-up actions: 3,418,789,194,096.
- Rule+ML Block/Hold transactions: 79,772.
- Rule-only Step-up/eKYC transactions: 8,367.
- ML-only Special Watchlist transactions: 21,016.

Data-driven insights:

- Root-cause concentration: Unauthorized transfer / capital outflow chiếm 69.7% số giao dịch High/Critical. Ý nghĩa: Review queue không nên xem toàn bộ case như nhau; nhánh nguyên nhân lớn nhất cần playbook riêng.
- Personalized IQR threshold matters: Giao dịch vượt ngưỡng IQR cá nhân hóa có High/Critical rate 60.8%, lift 57.7x so với nhóm không vượt. Ý nghĩa: Ngưỡng không còn là số cố định toàn ngân hàng; nó phản ánh thói quen riêng của từng khách hàng.
- Network exposure needs a second signal: Nhóm shared IP/device/beneficiary top 1% có High/Critical rate 7.7% so với baseline 7.7%; lift 1.0x. Ý nghĩa: Shared IP/device/beneficiary không nên tự động Block; cần kết hợp với cash-out, IQR breach hoặc repeated external transfer.
- Overdue is context, not a fraud label: Nhóm credit-risk group 3-5 có High/Critical rate 7.4%, nhóm 1-2 là 7.7%. Ý nghĩa: Overdue không tự kết luận fraud, nhưng giúp ưu tiên kiểm soát khi đi kèm cash-out hoặc IQR breach.
- Risk is time-patterned: Khung Mon lúc 16h có High/Critical rate 9.3%. Ý nghĩa: Step-up hoặc review staffing có thể ưu tiên khung giờ/ngày có risk-rate cao hơn.
- Hybrid matrix changes the banding logic: Rule+ML Block/Hold: 79,772; Rule-only Step-up/eKYC: 8,367; ML-only Watchlist: 21,016. Ý nghĩa: Risk band cuối được quyết định bằng ma trận vận hành Rule + ML để gắn trực tiếp với hành động kiểm soát.

Root-cause summary:

- Unauthorized transfer / capital outflow: 76,085 High/Critical giao dịch, risk trung bình 66.4/100.
- Account takeover / identity compromise: 32,510 High/Critical giao dịch, risk trung bình 43.1/100.
- AML network / mule-account pattern: 560 High/Critical giao dịch, risk trung bình 42.5/100.

Monthly stability backtest:

- Framework được kiểm tra theo từng tháng trong năm 2019.
- `model_metrics.json` có bảng monthly/quarterly stability gồm transaction count, average risk, P95 risk và High/Critical rate.
- Vì dữ liệu chỉ có năm 2019, đây là temporal robustness check, không phải crisis-period validation.

## 7. Vì sao có supervised model khi dữ liệu không có nhãn fraud thật?

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

## 8. Gợi ý vận hành thực tế

- Critical: Rule+ML cùng báo, Block/Hold near-real-time, gọi xác minh khách hàng, kiểm tra device/IP/beneficiary.
- High: Rule-only thì step-up/eKYC; ML-only thì Special Watchlist để bắt pattern mới.
- AML branch: escalation theo mạng lưới IP/device/beneficiary, không nhìn từng giao dịch riêng lẻ.
- Medium: IQR/model/rule early warning, theo dõi tăng cường và nâng cấp nếu lặp lại trong 7 ngày.
- KPI sau khi triển khai: prevention coverage, protected amount, hit rate trong top-K, false positive rate theo phân khúc, số case AML escalation, time-to-review.

## 9. Liên hệ với chuẩn nghiệp vụ quốc tế

- FATF Risk-Based Approach for Banking Sector: ngân hàng nên hiểu mức độ rủi ro, ưu tiên nguồn lực vào nơi rủi ro cao và áp dụng biện pháp giảm thiểu tương ứng. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Risk-based-approach-banking-sector.html
- FFIEC Authentication and Access Guidance: với digital banking, kiểm soát nên theo hướng layered security, MFA/step-up authentication và tăng kiểm soát khi giao dịch hoặc truy cập có rủi ro cao. Link: https://www.ffiec.gov/news/press-releases/2021/pr-08-11
- FATF Guidance on Risk-Based Supervision: tránh cách làm tick-box, tập trung vào full spectrum of risks và nơi rủi ro cao hơn. Link: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance-rba-supervision.html

Framework của dự án bám đúng tinh thần này: risk-based, layered controls, review queue theo capacity, và xAI reason codes để investigator kiểm tra được.

## 10. Hạn chế

Đây là framework prevention dùng weak label, không phải model xác nhận fraud tuyệt đối. Khi ngân hàng có kết quả review thật, cần đưa label đó quay lại pipeline để hiệu chỉnh rule, threshold, supervised model và đo Precision@K/Recall@K chính thức.
