# Data-Driven Insight Summary

## Root-cause concentration

- Evidence: Unauthorized transfer / capital outflow chiếm 69.7% số giao dịch High/Critical.
- Business meaning: Review queue không nên xem toàn bộ case như nhau; nhánh nguyên nhân lớn nhất cần playbook riêng.
- Figure: `root_cause_hybrid_heatmap.png`

## Personalized IQR threshold matters

- Evidence: Giao dịch vượt ngưỡng IQR cá nhân hóa có High/Critical rate 60.8%, lift 57.7x so với nhóm không vượt.
- Business meaning: Ngưỡng không còn là số cố định toàn ngân hàng; nó phản ánh thói quen riêng của từng khách hàng.
- Figure: `iqr_breach_lift.png`

## Network exposure needs a second signal

- Evidence: Nhóm shared IP/device/beneficiary top 1% có High/Critical rate 7.7% so với baseline 7.7%; lift 1.0x.
- Business meaning: Shared IP/device/beneficiary không nên tự động Block; cần kết hợp với cash-out, IQR breach hoặc repeated external transfer.
- Figure: `network_exposure_bubble.png`

## Overdue is context, not a fraud label

- Evidence: Nhóm credit-risk group 3-5 có High/Critical rate 7.4%, nhóm 1-2 là 7.7%.
- Business meaning: Overdue không tự kết luận fraud, nhưng giúp ưu tiên kiểm soát khi đi kèm cash-out hoặc IQR breach.
- Figure: `customer360_risk_heatmap.png`

## Risk is time-patterned

- Evidence: Khung Mon lúc 16h có High/Critical rate 9.3%.
- Business meaning: Step-up hoặc review staffing có thể ưu tiên khung giờ/ngày có risk-rate cao hơn.
- Figure: `time_risk_heatmap.png`

## Hybrid matrix changes the banding logic

- Evidence: Rule+ML Block/Hold: 79,772; Rule-only Step-up/eKYC: 8,367; ML-only Watchlist: 21,016.
- Business meaning: Risk band cuối được quyết định bằng ma trận vận hành Rule + ML để gắn trực tiếp với hành động kiểm soát.
- Figure: `hybrid_decision_matrix.png`