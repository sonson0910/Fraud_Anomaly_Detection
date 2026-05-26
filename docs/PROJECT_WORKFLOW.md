# Quy Trình Làm Việc Của Dự Án

Tài liệu này giải thích bài toán và cách source code hiện tại giải quyết bài toán Fraud & Anomaly Detection bằng dữ liệu thật của cuộc thi. Mục tiêu là để cả thành viên non-tech vẫn nắm được dự án đang làm gì, vì sao làm như vậy, và khi demo thì cần nhìn vào đâu.

## 1. Bài toán đang giải quyết

Cuộc thi đưa dữ liệu ngân hàng 360 độ gồm chân dung khách hàng, giao dịch, hoạt động digital banking và thông tin sản phẩm. Track 1 yêu cầu phát hiện gian lận hoặc bất thường như chiếm đoạt tài khoản, chuyển khoản trái phép và rửa tiền.

Điểm khó nhất là dữ liệu thật không có cột “fraud/not fraud”. Vì vậy dự án không thể claim nhãn fraud thật. Cách làm hiện tại là: phân tích nguyên nhân, tạo rule-based weak label, huấn luyện supervised prevention model từ weak label đó, rồi đưa ra hành động ngăn chặn như Allow, Monitor, Step-up Authentication hoặc Block/Hold.

## 2. Nguyên tắc quan trọng

- Không dùng synthetic data trong bản hiện tại.
- Không tự bịa confirmed fraud label.
- Có tạo `rule_fraud_label`, nhưng phải nói rõ đây là weak label do rule nghiệp vụ tạo ra, không phải nhãn fraud đã được investigator xác nhận.
- Không kết luận “đây chắc chắn là gian lận”; chỉ nói “giao dịch này cần Allow/Monitor/Step-up/Block vì các lý do sau”.
- Phân tích nguyên nhân trước, sau đó mới dùng rule để tạo label và huấn luyện model prevention.
- Mọi output quan trọng đều có reason codes để BGK và người nghiệp vụ hiểu được.

## 3. Ba nhánh nguyên nhân

### Account takeover / identity compromise

Nhánh này tìm tín hiệu tài khoản có thể bị người khác chiếm quyền:

- thiết bị mới sau khi khách hàng đã có lịch sử giao dịch,
- IP mới,
- giao dịch ngoài khung giờ thông thường,
- người thụ hưởng mới,
- hoạt động digital ở giai đoạn muộn trong cùng ngày.

### Unauthorized transfer / capital outflow

Nhánh này tập trung vào dòng tiền ra bất thường:

- chuyển khoản ra ngoài ngân hàng,
- số tiền cao hơn nhiều so với baseline của chính khách hàng,
- một ngày có số lượng giao dịch tăng đột biến,
- tổng tiền trong ngày tăng mạnh,
- số tiền chuyển ra lớn so với số dư CASA.

### AML network / mule-account pattern

Nhánh này tìm dấu hiệu mạng lưới, phù hợp với nghiệp vụ AML:

- nhiều khách hàng dùng chung IP,
- nhiều khách hàng dùng chung device,
- một người thụ hưởng nhận tiền từ nhiều khách hàng,
- giao dịch số tròn giá trị cao,
- nhiều chuyển khoản ra ngoài lặp lại.

## 4. Source code xử lý dữ liệu như thế nào

File chính là `src/fraud_pipeline.py`.

Luồng xử lý:

1. Đọc dữ liệu thật từ `Processed_Data/`.
2. Chuẩn hóa tên cột theo data dictionary. Ví dụ file thật có `TRANS_LV1`, trong dictionary ghi `TRXN_LV1`; pipeline hiểu đây là cùng một ý nghĩa.
3. Aggregate bảng activity rất lớn theo `CUSTOMER_NUMBER` và ngày.
4. Tạo baseline từng khách hàng: số tiền thường giao dịch, P95 số tiền, tần suất giao dịch/ngày, thiết bị/IP/người thụ hưởng đã từng thấy.
5. Ghép thêm thông tin sản phẩm theo tháng: deposit, lending, card.
6. Tách `Beneficiary_CUSTOMER_NUMBER = 0/0.0/NaN` thành non-customer/merchant beneficiary, không đưa nhóm này vào mạng lưới money mule như một khách hàng thật.
7. Tạo nhóm overdue tín dụng theo 5 bậc rủi ro dựa trên số ngày quá hạn.
8. Chấm điểm từng nhánh nguyên nhân bằng rule-based score.
9. Tạo `rule_fraud_label` từ rule score: nhóm fraud-by-rules, clean-by-rules và uncertain.
10. Huấn luyện supervised prevention model học từ weak label này.
11. Model trả `model_fraud_probability`, từ đó chia risk band: Low, Medium, High, Critical.
12. Gán hành động vận hành: Allow, Enhanced Monitoring, Step-up Authentication, Block/Hold.
13. Xuất reason codes và recommended action cho từng giao dịch.
14. Tạo monthly/quarterly stability backtest để kiểm tra prevention rate có ổn định theo thời gian không.
15. Chạy SHAP xAI engine để giải thích prevention score bằng surrogate tree model.

## 5. Vì sao vẫn dùng supervised model khi không có nhãn fraud thật?

Rule-based giúp giải thích rất tốt, nhưng nếu chỉ dùng rule thủ công thì khó mở rộng thành engine tự động cho giao dịch mới. Vì vậy pipeline dùng rule để tạo weak label trước.

Sau đó supervised model học từ weak label này để tự động hóa quyết định prevention. Model không thay thế nghiệp vụ; nó học lại logic nghiệp vụ trên nhiều feature hơn và trả xác suất rủi ro cho giao dịch mới.

Điểm cần nói rõ: metric của model hiện đo theo weak label, không phải confirmed fraud label.

Theo feedback mentor, vì title là giảm thiểu rủi ro nên mục tiêu chính của fraud track là bắt được càng nhiều giao dịch fraud càng tốt. Vì vậy model ưu tiên recall/prevention coverage trước, đồng thời vẫn theo dõi false-positive rate để không làm phiền khách hàng thông thường quá mức.

## 6. Xử lý Beneficiary và merchant

`Beneficiary_CUSTOMER_NUMBER = 0/0.0/NaN` không được xem là lỗi hệ thống mặc định. Trong nhiều loại giao dịch, đây là giao dịch với merchant hoặc tổ chức kinh doanh như telco, ví điện tử, QR, utility, credit/lending gateway. Vì vậy pipeline xử lý như sau:

- Không đưa 0/0.0/NaN vào mạng lưới customer-to-customer money mule.
- Tạo biến `beneficiary_is_customer` và `has_no_customer_beneficiary`.
- Với nhóm không có customer beneficiary, phân tích tiếp bằng `Merchant_ID_Masked` và `TRANS_LV2`.
- Nếu là merchant nội bộ/tín dụng mà không có customer beneficiary nhưng có tín hiệu bất thường, pipeline thêm reason code để review.

## 7. Overdue và bối cảnh rủi ro khách hàng

Overdue lending/credit được dùng như bối cảnh đánh giá khách hàng tốt/xấu. Pipeline tạo:

- `max_overdue_days`.
- `credit_risk_group_num` từ 1 đến 5.

Logic này giúp model hiểu rằng một giao dịch rủi ro đi kèm lịch sử quá hạn cao có thể cần kiểm soát chặt hơn, nhưng overdue không được dùng một mình để kết luận fraud.

## 8. SHAP/xAI giải thích gì?

xAI có hai lớp:

- Lớp nghiệp vụ: `top_reasons` giải thích trực tiếp vì sao giao dịch bị flag, ví dụ thiết bị mới, IP mới, chuyển khoản ngoài ngân hàng, số tiền lệch baseline.
- Lớp model: `src/xai_shap_engine.py` huấn luyện một tree surrogate để giải thích supervised prevention score, sau đó dùng SHAP TreeExplainer để chỉ ra feature nào kéo điểm rủi ro lên/xuống.

Nếu máy có đủ runtime cho XGBoost thì script có thể dùng XGBoost. Nếu thiếu `libomp` trên macOS, script tự fallback sang scikit-learn tree surrogate và vẫn xuất SHAP thật.

## 9. Các file output cần xem

- `outputs/transaction_risk_scores.csv`: bảng giao dịch đã scored, có risk band và prevention action.
- `outputs/customer_risk_summary.csv`: tổng hợp rủi ro theo khách hàng.
- `outputs/top_review_queue.csv`: 1,000 giao dịch đầu để demo nhanh.
- `outputs/root_cause_summary.csv`: tổng hợp số case theo ba nhánh nguyên nhân.
- `outputs/monthly_stability.csv`: kiểm tra stability theo tháng.
- `outputs/quarterly_stability.csv`: kiểm tra stability theo quý.
- `outputs/shap_feature_importance.csv`: global SHAP importance.
- `outputs/shap_local_explanations.csv`: local SHAP explanation cho top case.
- `outputs/model_metrics.json`: thông tin data quality, weak-label model metrics, prevention coverage và protected amount.
- `outputs/figures/`: chart dùng cho report/slide.
- `report/final_report_outline.md`: khung báo cáo cuối.
- `report/final_slide_deck.pdf`: slide deck PDF để nộp/trình bày.
- `report/final_slide_deck.pptx`: slide deck có thể chỉnh sửa.
- `notebooks/01_fraud_anomaly_detection.ipynb`: technical notebook.

## 10. Cách demo

Chạy toàn bộ pipeline:

```bash
python src/fraud_pipeline.py
```

Chạy SHAP xAI:

```bash
python src/xai_shap_engine.py
```

Tạo lại notebook:

```bash
python src/build_notebook.py
```

Tạo slide deck:

```bash
python src/build_slide_deck.py
```

Mở demo risk advisor:

```bash
python src/customer_risk_advisor.py --top-critical 3
```

Hoặc nhập một khách hàng cụ thể:

```bash
python src/customer_risk_advisor.py --customer-id <CUSTOMER_NUMBER>
```

Mở UI/chatbot demo:

```bash
streamlit run src/demo_app.py
```

Kết quả demo sẽ trả về risk band, prevention action, nhánh nguyên nhân chính, lý do bị flag và hành động đề xuất.

## 11. Cách đọc một dòng kết quả

Một giao dịch Critical không có nghĩa là “đã chắc chắn gian lận”. Nó có nghĩa là supervised prevention model, học từ weak labels của rule engine, đánh giá giao dịch này cần Block/Hold trước khi xử lý tiếp.

Các trường quan trọng:

- `risk_score_0_100`: điểm rủi ro tổng.
- `risk_band`: mức Low/Medium/High/Critical.
- `prevention_action`: Allow, Enhanced Monitoring, Step-up Authentication hoặc Block/Hold.
- `rule_fraud_label`: weak label do rule tạo ra.
- `model_fraud_probability`: xác suất do supervised prevention model dự báo.
- `primary_cause_branch`: nhánh nguyên nhân chính.
- `top_reasons`: lý do cụ thể.
- `recommended_action`: hành động đề xuất cho risk officer.

## 12. Điều cần nói rõ với BGK

Vì không có fraud label thật, dự án không claim Precision/Recall theo confirmed fraud. Thay vào đó, dự án báo performance, confusion matrix, recall, precision, false-positive rate và prevention coverage theo weak label, đồng thời nói rõ weak label là sản phẩm của rule engine.

- hiểu đúng assignment,
- phân tích nguyên nhân trước model,
- dùng dữ liệu thật,
- có framework prevention có thể vận hành,
- có xAI/reason code,
- có thể hiệu chỉnh sang confirmed-label supervised learning sau khi có feedback từ investigator.

## 13. Hướng nâng cấp nếu có thời gian

- Thêm graph/network visualization cho IP/device/beneficiary.
- Khi mentor/BTC cung cấp nhãn review đã xác minh, thay weak label bằng confirmed label và đo Precision@K/Recall@K chính thức.
- Thêm policy tuning theo review capacity của ngân hàng, ví dụ mỗi ngày chỉ review top 0.5% giao dịch.
