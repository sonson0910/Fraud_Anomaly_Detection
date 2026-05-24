# Quy Trình Làm Việc Của Dự Án

Tài liệu này giải thích bài toán và cách source code hiện tại giải quyết bài toán Fraud & Anomaly Detection bằng dữ liệu thật của cuộc thi. Mục tiêu là để cả thành viên non-tech vẫn nắm được dự án đang làm gì, vì sao làm như vậy, và khi demo thì cần nhìn vào đâu.

## 1. Bài toán đang giải quyết

Cuộc thi đưa dữ liệu ngân hàng 360 độ gồm chân dung khách hàng, giao dịch, hoạt động digital banking và thông tin sản phẩm. Track 1 yêu cầu phát hiện gian lận hoặc bất thường như chiếm đoạt tài khoản, chuyển khoản trái phép và rửa tiền.

Điểm khó nhất là dữ liệu thật không có cột “fraud/not fraud”. Vì vậy dự án không thể làm supervised learning kiểu học từ nhãn đúng/sai. Thay vào đó, dự án xây một framework xếp hạng rủi ro: giao dịch nào đáng nghi hơn thì được đưa lên đầu hàng đợi để risk officer kiểm tra.

## 2. Nguyên tắc quan trọng

- Không dùng synthetic data trong bản hiện tại.
- Không tự bịa fraud label.
- Không kết luận “đây chắc chắn là gian lận”; chỉ nói “giao dịch này đáng review vì các lý do sau”.
- Phân tích nguyên nhân trước, sau đó mới dùng model để lượng hóa bất thường.
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
6. Chấm điểm từng nhánh nguyên nhân bằng rule-based score.
7. Dùng Isolation Forest để bắt giao dịch lệch khỏi phân bố chung.
8. Gộp điểm cuối: 60% rule score và 40% anomaly model score.
9. Chia risk band: Low, Medium, High, Critical.
10. Xuất reason codes và recommended action cho từng giao dịch.
11. Tạo monthly/quarterly stability backtest để kiểm tra review-rate có ổn định theo thời gian không.
12. Chạy SHAP xAI engine để giải thích risk score bằng surrogate tree model.

## 5. Vì sao vẫn dùng model khi không có nhãn?

Rule-based giúp giải thích rất tốt nhưng có thể bỏ sót các pattern lạ. Isolation Forest là mô hình unsupervised thường dùng cho anomaly detection khi chưa có label. Nó không cần biết trước đâu là fraud; nó học phân bố bình thường của dữ liệu và đánh điểm cao hơn cho các điểm lệch.

Trong dự án này, model không thay thế nghiệp vụ. Model chỉ bổ sung lớp phát hiện bất thường. Phần giải thích vẫn dựa trên root-cause rules để người chấm và risk officer đọc được.

## 6. SHAP/xAI giải thích gì?

xAI có hai lớp:

- Lớp nghiệp vụ: `top_reasons` giải thích trực tiếp vì sao giao dịch bị flag, ví dụ thiết bị mới, IP mới, chuyển khoản ngoài ngân hàng, số tiền lệch baseline.
- Lớp model: `src/xai_shap_engine.py` huấn luyện một tree surrogate để bắt chước `risk_score_0_100`, sau đó dùng SHAP TreeExplainer để chỉ ra feature nào kéo điểm rủi ro lên/xuống.

Nếu máy có đủ runtime cho XGBoost thì script có thể dùng XGBoost. Nếu thiếu `libomp` trên macOS, script tự fallback sang scikit-learn tree surrogate và vẫn xuất SHAP thật.

## 7. Các file output cần xem

- `outputs/transaction_risk_scores.csv`: bảng giao dịch đã scored, sắp xếp từ rủi ro cao xuống thấp.
- `outputs/customer_risk_summary.csv`: tổng hợp rủi ro theo khách hàng.
- `outputs/top_review_queue.csv`: 1,000 giao dịch đầu để demo nhanh.
- `outputs/root_cause_summary.csv`: tổng hợp số case theo ba nhánh nguyên nhân.
- `outputs/monthly_stability.csv`: kiểm tra stability theo tháng.
- `outputs/quarterly_stability.csv`: kiểm tra stability theo quý.
- `outputs/shap_feature_importance.csv`: global SHAP importance.
- `outputs/shap_local_explanations.csv`: local SHAP explanation cho top case.
- `outputs/model_metrics.json`: thông tin data quality, thresholds, phân phối risk band.
- `outputs/figures/`: chart dùng cho report/slide.
- `report/final_report_outline.md`: khung báo cáo cuối.
- `report/final_slide_deck.pdf`: slide deck PDF để nộp/trình bày.
- `report/final_slide_deck.pptx`: slide deck có thể chỉnh sửa.
- `notebooks/01_fraud_anomaly_detection.ipynb`: technical notebook.

## 8. Cách demo

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

Kết quả demo sẽ trả về risk band, nhánh nguyên nhân chính, lý do bị flag và hành động đề xuất.

## 9. Cách đọc một dòng kết quả

Một giao dịch Critical không có nghĩa là “đã chắc chắn gian lận”. Nó có nghĩa là giao dịch này nằm trong top rủi ro theo framework hiện tại và cần được review trước.

Các trường quan trọng:

- `risk_score_0_100`: điểm rủi ro tổng.
- `risk_band`: mức Low/Medium/High/Critical.
- `primary_cause_branch`: nhánh nguyên nhân chính.
- `top_reasons`: lý do cụ thể.
- `recommended_action`: hành động đề xuất cho risk officer.

## 10. Điều cần nói rõ với BGK

Vì không có fraud label thật, dự án không báo Precision/Recall giả. Đó là điểm mạnh về tính trung thực. Thay vào đó, dự án chứng minh:

- hiểu đúng assignment,
- phân tích nguyên nhân trước model,
- dùng dữ liệu thật,
- có framework có thể vận hành,
- có xAI/reason code,
- có thể mở rộng sang supervised learning sau khi có feedback từ investigator.

## 11. Hướng nâng cấp nếu có thời gian

- Thêm graph/network visualization cho IP/device/beneficiary.
- Khi mentor/BTC cung cấp nhãn review, thêm supervised model và đo Precision@K/Recall@K chính thức.
- Thêm policy tuning theo review capacity của ngân hàng, ví dụ mỗi ngày chỉ review top 0.5% giao dịch.
