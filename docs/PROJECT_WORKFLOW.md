# Quy trình làm việc hiện tại của dự án

Tài liệu này giải thích dự án đang làm gì, vì sao làm như vậy, các file chính có vai trò gì, và team cần chạy lại pipeline như thế nào.

## 1. Dự án đang giải quyết bài toán gì?

Dự án này phục vụ bài toán 1 của G'Contest 2026: **Fraud & Anomaly Detection - phát hiện gian lận và bất thường trong giao dịch ngân hàng số**.

Vì dữ liệu thật chưa có trong workspace, dự án hiện dùng **synthetic data** được sinh theo đúng data dictionary của BTC. Synthetic data không nhằm giả làm dữ liệu thật, mà dùng để chứng minh toàn bộ quy trình phân tích từ đầu đến cuối:

1. Hiểu cấu trúc dữ liệu ngân hàng.
2. Xác định nguyên nhân rủi ro trước khi xây model.
3. Sinh dữ liệu có hành vi bình thường và bất thường có kiểm soát.
4. Tạo feature, rule score, anomaly score.
5. Xuất kết quả risk score, explainability, report và demo.

Luồng kể chuyện chính của dự án là **cause-first**, tức là không bắt đầu bằng model. Team trước hết xác định các nguyên nhân có thể dẫn đến fraud/anomaly, sau đó mới xây framework phát hiện.

## 2. Ba nhóm nguyên nhân chính

Framework hiện chia nguyên nhân rủi ro thành 3 nhánh:

1. **Account access / identity compromise**
   - Dấu hiệu: thiết bị mới, IP mới, OTP request, add beneficiary, change password.
   - Ý nghĩa nghiệp vụ: có khả năng tài khoản bị chiếm đoạt hoặc bị truy cập bất thường.

2. **Abnormal transaction behavior**
   - Dấu hiệu: giao dịch ngoài giờ quen thuộc, amount vượt baseline, tần suất tăng đột biến, dòng tiền ra lớn so với số dư.
   - Ý nghĩa nghiệp vụ: hành vi giao dịch lệch khỏi lịch sử bình thường của khách hàng.

3. **Network / AML linkage**
   - Dấu hiệu: IP/device dùng chung nhiều khách hàng, chuyển khoản ngoài hệ thống, giao dịch số tròn lặp lại.
   - Ý nghĩa nghiệp vụ: có khả năng liên quan tới money mule, proxy account hoặc laundering pattern.

Ba nhánh này giúp report và demo giải thích rõ: **giao dịch bị đánh dấu vì nguyên nhân gì**, không chỉ vì model trả điểm cao.

## 3. Dữ liệu trong dự án

Synthetic data được sinh vào `data/raw/`, gồm 6 bảng giống data dictionary của BTC:

- `Data_Customer.csv`: chân dung khách hàng.
- `Data_Transaction.csv`: lịch sử giao dịch e-banking.
- `Data_Activity.csv`: hành vi số theo thời gian.
- `Data_Deposit.csv`: tiền gửi.
- `Data_Lending.csv`: tín dụng.
- `Data_Card.csv`: thẻ.

Ngoài ra có `synthetic_ground_truth.csv`, chứa nhãn anomaly synthetic để kiểm tra mô hình. Nhãn này chỉ dùng cho evaluation, không nằm trong schema gốc của BTC.

Một số điểm quan trọng:

- `CUSTOMER_NUMBER` là key nối giữa các bảng.
- `ACTIVITY_NO` được hiểu như thứ tự hành vi: số nhỏ là hành động sớm, số lớn là hành động sau hoặc nhạy cảm hơn.
- Dữ liệu trải từ năm 2019 đến 2026 để kiểm tra tính ổn định qua các giai đoạn thời gian.
- Các CSV lớn trong `data/raw/` không được commit lên GitHub vì có thể tái sinh bằng script.

## 4. Vai trò các file chính

### `src/generate_synthetic_data.py`

Sinh toàn bộ synthetic data.

Script này tạo:

- 3,000 khách hàng.
- 150,000 giao dịch.
- 220,000 activity logs.
- Monthly product snapshots cho deposit, lending, card.
- 2,700 giao dịch anomaly được cài nhãn.

Các anomaly được inject theo các pattern như account takeover, unauthorized transfer, money laundering proxy và behavioral outlier.

### `src/fraud_pipeline.py`

Đây là pipeline chính.

Script này làm các bước:

1. Load dữ liệu raw.
2. Validate schema.
3. Feature engineering.
4. Tính root-cause rule score.
5. Chạy Isolation Forest để phát hiện điểm bất thường đa chiều.
6. Kết hợp thành risk score cuối cùng.
7. Xuất transaction risk score, customer summary, root-cause summary, metrics, figures và report outline.

Risk score cuối cùng hiện dùng công thức:

```text
risk_score = 60% model_score + 40% rule_score
```

Lý do dùng công thức hybrid: rule score dễ giải thích, model score bắt được pattern đa chiều hơn.

### `src/build_notebook.py`

Tạo lại notebook kỹ thuật ở:

```text
notebooks/01_fraud_anomaly_detection.ipynb
```

Notebook dùng để nộp technical notebook cho BGK. Nội dung gồm setup, sinh dữ liệu, schema validation, EDA, root-cause hypothesis, feature engineering, model, evaluation, explainability và recommendation.

### `src/customer_risk_advisor.py`

Demo dạng risk advisor.

Script này đọc output đã có và giải thích rủi ro theo `CUSTOMER_NUMBER` hoặc `transaction_row_id`. Đây là phần có thể dùng để demo live hoặc quay video:

```bash
python src/customer_risk_advisor.py --top-critical 3
python src/customer_risk_advisor.py --customer-id CUS000001
python src/customer_risk_advisor.py --transaction-id TRX0000001
```

Ý tưởng mở rộng: nếu có API LLM, có thể biến phần này thành chatbot giải thích cho risk officer. LLM không cần là lõi phát hiện fraud; nó phù hợp hơn ở lớp giải thích và hỏi đáp.

## 5. Output chính

Sau khi chạy pipeline, các output quan trọng nằm trong `outputs/`:

- `transaction_risk_scores.csv`: điểm rủi ro từng giao dịch.
- `customer_risk_summary.csv`: tổng hợp rủi ro theo khách hàng.
- `root_cause_summary.csv`: tóm tắt theo 3 nhánh nguyên nhân.
- `model_metrics.json`: metrics mô hình.
- `figures/*.png`: biểu đồ cho report.

Report nằm trong `report/`:

- `final_report_outline.md`: report outline tiếng Việt.
- `final_report_outline.pdf`: bản PDF xuất từ markdown.

Notebook nằm trong `notebooks/`.

## 6. Quy trình chạy lại từ đầu

Từ thư mục project, tạo môi trường và cài thư viện:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy lại toàn bộ pipeline:

```bash
python src/generate_synthetic_data.py
python src/fraud_pipeline.py
python src/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/01_fraud_anomaly_detection.ipynb --ExecutePreprocessor.timeout=900
```

Kiểm tra:

```bash
python -m pytest -q
```

Chạy demo:

```bash
python src/customer_risk_advisor.py --top-critical 1
```

## 7. Những file không nên commit

Các file/thư mục sau không đưa lên GitHub:

- `.venv/`
- `.pytest_cache/`
- `__pycache__/`
- `data/raw/*.csv`
- `outputs/transaction_risk_scores.csv`
- `outputs/customer_risk_summary.csv`

Lý do: các file này nặng hoặc có thể tái sinh. Repo chỉ cần giữ source code, notebook, report, figures nhẹ và metrics summary.

## 8. Cách giải thích dự án với BGK

Thông điệp nên dùng:

> Vì dữ liệu không có nhãn fraud, nhóm không bắt đầu bằng supervised AI. Nhóm xây dựng risk taxonomy trước, tạo baseline hành vi khách hàng, phát hiện lệch chuẩn bằng rule + unsupervised anomaly detection, sau đó dùng xAI để giải thích từng cảnh báo.

Điểm mạnh cần nhấn:

- Có logic nguyên nhân trước khi có model.
- Có 3 nhánh rủi ro rõ ràng, gắn với nghiệp vụ ngân hàng.
- Không phụ thuộc vào label fraud.
- Có kiểm tra stability theo thời gian từ 2019 đến 2026.
- Có explainability ở cấp giao dịch.
- Có demo advisor để nhập khách hàng/giao dịch và nhận giải thích.

## 9. Việc team có thể làm tiếp

- Chuyển report outline thành slide PDF đẹp hơn.
- Quay video demo `customer_risk_advisor.py`.
- Thêm dashboard Streamlit nếu còn thời gian.
- Khi có dữ liệu thật, thay synthetic data bằng dữ liệu BTC, giữ nguyên pipeline feature/model/report.
- Nếu có label fraud thật, bổ sung supervised benchmark như XGBoost/LightGBM và SHAP.
