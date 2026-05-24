# Dự án này đang làm gì và source code giải quyết bài toán như thế nào?

Tài liệu này dành cho cả thành viên non-tech và tech trong team. Mục tiêu là để mọi người hiểu cùng một câu chuyện: **bài toán là gì, vì sao khó, dự án đang giải quyết theo logic nào, kết quả đọc ra sao, và source code nào làm phần nào**.

## 1. Bài toán nói bằng ngôn ngữ đời thường

Trong ngân hàng số, mỗi ngày có rất nhiều khách hàng đăng nhập, kiểm tra số dư, thêm người nhận tiền, chuyển khoản, thanh toán hóa đơn, dùng thẻ, vay, gửi tiết kiệm. Phần lớn hoạt động là bình thường. Nhưng trong đó có thể có các tình huống rủi ro như:

- Tài khoản bị người khác chiếm quyền và chuyển tiền đi.
- Khách hàng đột nhiên chuyển khoản số tiền rất lớn so với thói quen trước đây.
- Một thiết bị hoặc IP xuất hiện ở nhiều tài khoản khác nhau.
- Nhiều giao dịch nhỏ/lặp lại được dùng để che giấu dòng tiền.
- Khách hàng có nhiều hành động nhạy cảm trước giao dịch, ví dụ OTP request, add beneficiary, change password.

Bài toán của nhóm là xây một framework để trả lời:

> Giao dịch hoặc khách hàng nào đang bất thường, vì sao bất thường, mức độ rủi ro là bao nhiêu, và ngân hàng nên làm gì tiếp theo?

Điểm quan trọng: dữ liệu BTC chưa có nhãn rõ ràng kiểu “fraud” hoặc “not fraud”. Vì vậy nếu cố làm supervised AI ngay từ đầu sẽ không hợp lý. Dự án chọn hướng thực tế hơn: **hiểu nguyên nhân trước, tạo baseline hành vi, phát hiện lệch chuẩn, rồi giải thích kết quả**.

## 2. Tư duy chính của dự án: không bắt đầu từ model

Nhiều bài thi dễ sa vào hướng “chọn model gì?”. Nhưng với fraud/anomaly detection, BGK thường đánh giá cao việc hiểu **nguyên nhân rủi ro** trước.

Vì vậy dự án đi theo luồng:

```text
Dữ liệu ngân hàng
-> Hiểu hành vi bình thường của khách hàng
-> Xác định nhóm nguyên nhân rủi ro
-> Tạo feature đo các dấu hiệu đó
-> Chấm điểm bằng rule + anomaly model
-> Giải thích vì sao bị cảnh báo
-> Đề xuất hành động nghiệp vụ
```

Nói ngắn gọn: **model chỉ là một phần của hệ thống**, không phải toàn bộ bài giải.

## 3. Ba nhóm nguyên nhân rủi ro

Framework hiện chia rủi ro thành 3 nhóm dễ hiểu.

### Nhóm 1: Rủi ro truy cập tài khoản

Tên trong output: `Account access / identity compromise`

Ý nghĩa: có thể tài khoản bị truy cập bởi người lạ hoặc bị chiếm quyền.

Dấu hiệu ví dụ:

- Thiết bị mới.
- IP mới.
- Có OTP request bất thường.
- Thêm người nhận mới.
- Đổi mật khẩu hoặc cập nhật profile gần thời điểm giao dịch.

Ví dụ giải thích cho BGK:

> Giao dịch này đáng nghi vì trước đó tài khoản có hành vi nhạy cảm, sau đó dùng thiết bị/IP mới để thực hiện giao dịch giá trị cao.

### Nhóm 2: Rủi ro hành vi giao dịch bất thường

Tên trong output: `Abnormal transaction behavior`

Ý nghĩa: giao dịch lệch mạnh khỏi thói quen trước đây của chính khách hàng.

Dấu hiệu ví dụ:

- Giao dịch lúc đêm/khoảng giờ không quen thuộc.
- Số tiền vượt xa mức bình thường của khách hàng.
- Tần suất giao dịch tăng đột biến trong một ngày.
- Dòng tiền ra lớn so với số dư bình quân.

Ví dụ giải thích:

> Không phải cứ số tiền lớn là gian lận. Điều đáng nghi là số tiền này lớn bất thường so với chính lịch sử của khách hàng.

### Nhóm 3: Rủi ro mạng lưới và AML

Tên trong output: `Network / AML linkage`

Ý nghĩa: có thể liên quan tới tài khoản trung gian, money mule, hoặc rửa tiền.

Dấu hiệu ví dụ:

- Một IP dùng bởi nhiều khách hàng.
- Một thiết bị dùng bởi nhiều khách hàng.
- Chuyển khoản ngoài hệ thống.
- Giao dịch số tròn, lặp lại, hoặc có pattern dòng tiền ra.

Ví dụ giải thích:

> Giao dịch không chỉ bất thường riêng lẻ mà còn nằm trong mạng lưới IP/device có liên hệ với nhiều tài khoản, nên cần ưu tiên kiểm tra AML.

## 4. Dữ liệu trong dự án

Dữ liệu thật chưa có trong workspace, nên dự án sinh **synthetic data** dựa trên data dictionary của BTC.

Synthetic data nghĩa là dữ liệu giả lập có kiểm soát. Nó không dùng để khẳng định kết quả kinh doanh thật, mà dùng để chứng minh framework có thể chạy end-to-end.

Dự án sinh 6 bảng giống cấu trúc BTC:

- `Data_Customer.csv`: thông tin/chân dung khách hàng.
- `Data_Transaction.csv`: giao dịch e-banking.
- `Data_Activity.csv`: hành vi số, ví dụ login, OTP request, add beneficiary.
- `Data_Deposit.csv`: tiền gửi.
- `Data_Lending.csv`: tín dụng.
- `Data_Card.csv`: thẻ.

Bảng nối chính là `CUSTOMER_NUMBER`.

Dự án cũng sinh một file kiểm chứng riêng:

- `synthetic_ground_truth.csv`: đánh dấu giao dịch nào là anomaly synthetic.

File ground truth này chỉ dùng để đánh giá mô hình trong môi trường giả lập. Khi có dữ liệu thật, phần này sẽ được thay bằng nhãn thật nếu BTC cung cấp hoặc bằng kết quả manual review.

## 5. Dự án tạo dữ liệu bất thường như thế nào?

Source code cố tình cài vào dữ liệu một số tình huống rủi ro thường gặp:

- `ACCOUNT_TAKEOVER`: tài khoản bị chiếm quyền, dùng thiết bị/IP mới, giao dịch lúc giờ lạ.
- `UNAUTHORIZED_TRANSFER`: chuyển khoản trái phép hoặc khác thói quen.
- `MONEY_LAUNDERING_PROXY`: pattern giống tài khoản trung gian/rửa tiền.
- `BEHAVIORAL_OUTLIER`: hành vi đột ngột khác baseline.

Nhờ vậy, khi pipeline chạy xong, team có thể kiểm tra xem framework có bắt đúng các tình huống bất thường đã cài vào hay không.

## 6. Source code xử lý bài toán theo các bước nào?

### Bước 1: Sinh dữ liệu

File phụ trách: `src/generate_synthetic_data.py`

File này tạo dữ liệu từ năm 2019 đến 2026, gồm:

- 3,000 khách hàng.
- 150,000 giao dịch.
- 220,000 activity logs.
- Dữ liệu tháng về tiền gửi, vay và thẻ.
- 2,700 giao dịch anomaly synthetic.

Kết quả được ghi vào `data/raw/`.

### Bước 2: Kiểm tra dữ liệu

File phụ trách: `src/fraud_pipeline.py`

Pipeline kiểm tra mỗi bảng có đủ cột như data dictionary không, có thiếu dữ liệu không, có duplicate không. Mục tiêu là đảm bảo bài phân tích không xây trên dữ liệu sai cấu trúc.

### Bước 3: Tạo feature

Feature là các biến dùng để mô tả hành vi.

Ví dụ:

- Giao dịch có xảy ra ban đêm không?
- Số tiền có vượt mức P95 lịch sử của khách hàng không?
- Hôm đó khách hàng có nhiều hoạt động nhạy cảm không?
- Thiết bị/IP này có dùng bởi nhiều khách hàng không?
- Số tiền giao dịch có lớn so với số dư bình quân không?

Đây là bước chuyển từ dữ liệu thô sang tín hiệu nghiệp vụ.

### Bước 4: Chấm điểm bằng rule

Rule là các luật dễ hiểu.

Ví dụ:

```text
Nếu giao dịch xảy ra ban đêm + số tiền cao bất thường + thiết bị mới
=> tăng điểm rủi ro
```

Rule giúp hệ thống giải thích được vì sao giao dịch bị cảnh báo. Đây là phần rất quan trọng với fraud/risk, vì ngân hàng không thể chỉ nói “model bảo vậy”.

### Bước 5: Chấm điểm bằng anomaly model

Dự án dùng `Isolation Forest`, một mô hình phát hiện điểm bất thường khi không có label fraud thật.

Hiểu đơn giản:

> Model học hình dạng chung của hành vi bình thường. Những giao dịch quá khác đám đông hoặc khác baseline sẽ bị đẩy điểm rủi ro lên cao.

Mô hình này không thay thế rule. Nó bổ sung thêm góc nhìn đa chiều mà rule đơn lẻ có thể bỏ sót.

### Bước 6: Kết hợp thành risk score cuối cùng

Risk score cuối cùng hiện được tính theo công thức:

```text
risk_score = 60% model_score + 40% rule_score
```

Sau đó hệ thống gán nhãn:

- `Low`: rủi ro thấp.
- `Medium`: cần theo dõi.
- `High`: nên step-up authentication hoặc review.
- `Critical`: nên giữ giao dịch/manual review/AML escalation.

### Bước 7: Giải thích kết quả

Mỗi giao dịch được xuất ra:

- `risk_score_0_100`: điểm rủi ro.
- `risk_band`: Low/Medium/High/Critical.
- `primary_cause_branch`: nhánh nguyên nhân chính.
- `top_reasons`: các lý do cụ thể.
- `recommended_action`: hành động đề xuất.

Ví dụ một kết quả có thể đọc như sau:

> Giao dịch của khách hàng CUS002645 bị xếp Critical vì thuộc nhóm Network / AML linkage, xảy ra lúc 23h, số tiền cao bất thường, dòng tiền ra lớn so với số dư, dùng thiết bị mới, và model anomaly cũng đánh giá tổng thể cao. Hành động đề xuất là giữ giao dịch để manual review và AML escalation nếu pattern lặp lại.

## 7. Các output chính dùng để làm bài

Sau khi chạy pipeline, các file quan trọng là:

- `outputs/transaction_risk_scores.csv`: bảng chấm điểm từng giao dịch.
- `outputs/customer_risk_summary.csv`: bảng tổng hợp rủi ro theo khách hàng.
- `outputs/root_cause_summary.csv`: bảng tóm tắt theo 3 nhóm nguyên nhân.
- `outputs/model_metrics.json`: metrics mô hình.
- `outputs/figures/*.png`: biểu đồ dùng cho report.
- `report/final_report_outline.md`: nội dung report dạng markdown.
- `report/final_report_outline.pdf`: bản PDF của report outline.
- `notebooks/01_fraud_anomaly_detection.ipynb`: technical notebook.

## 8. Demo hoạt động như thế nào?

File demo:

```text
src/customer_risk_advisor.py
```

Demo đọc kết quả đã được pipeline tạo ra và trả lời theo kiểu risk advisor.

Ví dụ:

```bash
python src/customer_risk_advisor.py --top-critical 1
```

Kết quả demo sẽ cho biết:

- Giao dịch nào bị flag.
- Thuộc khách hàng nào.
- Risk band và risk score.
- Nguyên nhân chính.
- Vì sao bị cảnh báo.
- Ngân hàng nên làm gì.

Đây là phần có thể dùng để quay video hoặc demo live. Nếu muốn phát triển thêm, có thể biến demo này thành chatbot/LLM assistant để risk officer hỏi đáp tự nhiên hơn.

## 9. Người non-tech nên nhớ điều gì?

Nếu cần giải thích dự án trong 30 giây:

> Dự án xây một hệ thống phát hiện giao dịch bất thường cho ngân hàng số. Vì dữ liệu không có nhãn gian lận, nhóm không dùng supervised AI ngay. Thay vào đó, nhóm phân tích các nguyên nhân rủi ro trước, tạo baseline hành vi khách hàng, dùng rule và anomaly model để chấm điểm, rồi giải thích rõ vì sao từng giao dịch bị cảnh báo và ngân hàng nên xử lý thế nào.

Nếu cần giải thích trong 3 ý:

1. Dự án không chỉ dự đoán, mà còn giải thích nguyên nhân.
2. Framework có 3 nhóm rủi ro: truy cập tài khoản, hành vi giao dịch, mạng lưới/AML.
3. Kết quả cuối cùng là risk score, reason codes và recommended action.

## 10. Người tech cần biết gì để chạy lại?

Cài môi trường:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy lại toàn bộ:

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

## 11. Những file không đưa lên GitHub

Một số file lớn có thể tái sinh nên không commit:

- `data/raw/*.csv`
- `outputs/transaction_risk_scores.csv`
- `outputs/customer_risk_summary.csv`
- `.venv/`
- cache như `.pytest_cache/`, `__pycache__/`

Repo GitHub chỉ giữ source code, notebook, report, figures nhẹ và metrics summary.

## 12. Nếu có dữ liệu thật thì làm gì tiếp?

Khi BTC cung cấp dữ liệu thật, team nên:

1. Thay phần synthetic data bằng dữ liệu thật.
2. Giữ lại pipeline feature engineering và scoring.
3. Recalibrate threshold Low/Medium/High/Critical.
4. Nếu có label fraud thật, thêm supervised model như XGBoost/LightGBM.
5. Dùng SHAP hoặc feature importance để tăng phần xAI.
6. Cập nhật report bằng insight từ dữ liệu thật.

Nói cách khác, synthetic data chỉ là bản mô phỏng để chứng minh framework; phần cốt lõi cần giữ là logic nguyên nhân, feature, scoring, explainability và action recommendation.
