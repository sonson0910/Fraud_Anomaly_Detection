# Feedback Verification - Fraud Prevention Flow

Tài liệu này đối chiếu feedback mentor với implementation hiện tại để tránh nhầm lẫn với flow cũ.

## Kết luận nhanh

Source hiện tại không còn đi theo hướng cũ `Isolation Forest + 60/40 hybrid score`. Flow hiện tại đã được đổi thành đúng 4 giai đoạn và 11 bước:

1. Data Cleaning: đọc dữ liệu thật, chuẩn hóa schema, xử lý beneficiary/merchant, aggregate activity log và ghép product snapshots.
2. Four Baseline Metrics: Transactional, Financial, Environmental, Behavioral.
3. Customer 360 Feature Extraction: rolling window 30/60/90 ngày và một dòng baseline cho mỗi khách hàng.
4. Baseline EDA: sinh insight/figure từ lift, heatmap, network exposure và Customer 360, không chỉ biểu đồ cột đơn giản.
5. IQR Thresholding: `Q3 + 1.5 * IQR` ở cấp khách hàng/ngày/giao dịch.
6. Dynamic Rule Engine: ba nhánh nguyên nhân Account Takeover, Unauthorized Transfer, AML/Mule Network.
7. Risk-Scoring & Auto-Labeling: Rule-based weak label được tạo qua `rule_fraud_label` để dùng cho weak supervision.
8. ML Training: RandomForest prevention model học từ weak label, tối ưu recall/precision theo weak label.
9. Hybrid Matrix: Rule+ML = Block/Hold, Rule-only = Step-up/eKYC, ML-only = Watchlist, No-alert = Allow.
10. Dashboard: Streamlit hiển thị prevention impact, protected amount, Customer 360, insight và case review.
11. xAI: reason codes + SHAP surrogate giải thích final hybrid prevention score.

## Verification Matrix

| Feedback | Đã đáp ứng ở đâu | Trạng thái |
|---|---|---|
| Phân tích nguyên nhân trước, sau đó mới framework/model | `src/fraud_pipeline.py` tạo 3 branch score trước khi train model; notebook section Cause-First Fraud Hypotheses; slide Why Cause-First | Done |
| Có 3 nhánh problem chính | `branch_account_takeover_score`, `branch_unauthorized_transfer_score`, `branch_aml_network_score` | Done |
| Không tự bịa confirmed fraud vì data không có label | `rule_fraud_label` được ghi rõ là weak label; `model_metrics.json` có `ground_truth_available=false` | Done |
| Rule-based tạo label trước rồi mới train model | `build_rule_score()` tạo `rule_fraud_label`; `train_prevention_model()` train RandomForest từ label này | Done |
| Ưu tiên bắt fraud/rủi ro, nhưng vẫn theo dõi false positive | metrics có recall, precision, false-positive rate, confusion matrix; dashboard Prevention impact | Done |
| Customer 360/micro-view | `outputs/customer_360_baseline.csv`; dashboard tab Customer 360 | Done |
| Rolling window 30/60/90 ngày | feature `rolling_30d_*`, `rolling_60d_*`, `rolling_90d_*`; metrics `baseline_engineering.rolling_windows_days` | Done |
| IQR threshold `Q3 + 1.5*IQR` | `_iqr_upper()`, `amount_vs_iqr_upper`, `daily_count_vs_iqr_upper`, `daily_amount_vs_iqr_upper` | Done |
| Beneficiary 0/0.0/NaN không mặc định là missing | `beneficiary_is_customer`, `has_no_customer_beneficiary`, `Merchant_ID_Masked` flags; non-customer beneficiary không đưa vào money mule network | Done |
| Merchant/internal credit không có customer beneficiary có thể là risk context | reason code `Merchant nội bộ/tín dụng không có customer beneficiary...` | Done |
| Overdue/nhóm nợ là bối cảnh rủi ro, không phải fraud label độc lập | `max_overdue_days`, `credit_risk_group_num`; reason code chỉ là context bổ sung | Done |
| Tính giải thích và ổn định quan trọng hơn black-box | reason codes + SHAP surrogate + monthly/quarterly stability | Done |
| Hybrid matrix Rule + ML | `hybrid_decision`: Rule+ML, Rule-only, ML-only, No-alert | Done |
| Output vận hành, không chỉ phát hiện | `prevention_action`, `recommended_action`, `protected_amount`, dashboard Prevention impact | Done |
| Dashboard thể hiện chặn/ngăn ngừa bao nhiêu | metrics `blocked_transactions`, `step_up_transactions`, `protected_amount_block_or_step_up`; Streamlit tab Prevention impact | Done |
| xAI base trên report/model | `src/xai_shap_engine.py`, `outputs/shap_feature_importance.csv`, `outputs/shap_local_explanations.csv`, SHAP tab | Done |
| Biểu đồ phải có insight, không chỉ bar chart đơn giản | `outputs/insight_summary.csv`; figures `root_cause_hybrid_heatmap`, `iqr_breach_lift`, `time_risk_heatmap`, `network_exposure_bubble`, `customer360_risk_heatmap` | Done |
| Risk band không còn chia theo score/quantile cũ | `risk_band_policy` trong `model_metrics.json`; final band đi sau `hybrid_decision` | Done |
| SHAP giải thích decision cuối, không phải score cũ | `src/xai_shap_engine.py` target final hybrid `risk_score_0_100`; `shap_metrics.json` ghi rõ | Done |
| Không dùng flow synthetic/Isolation Forest cũ | Không có `src/generate_synthetic_data.py`; không có `synthetic_ground_truth.csv`; không còn Isolation Forest trong source | Verified |

## File quan trọng để BGK kiểm tra

- `notebooks/01_fraud_anomaly_detection.ipynb`
- `src/fraud_pipeline.py`
- `src/demo_app.py`
- `outputs/model_metrics.json`
- `outputs/root_cause_summary.csv`
- `outputs/shap_feature_importance.csv`
- `report/final_slide_deck.pdf`
- `docs/PROJECT_WORKFLOW.md`
