# Báo cáo Phân tích Chuyên sâu — GPU FinOps Optimization (NimbusAI)

- **Học viên:** Nguyễn Tuấn Anh
- **Mã sinh viên:** 2A202602535
- **Đề tài:** Cắt giảm chi phí GPU & Đơn giá Token (`$/1M-token`) tại NimbusAI
- **Thời điểm dữ liệu:** Tháng 6/2026 (Mô phỏng tất định, seed=25)

---

## 1. So sánh Tổng quan: Baseline vs. Optimized

Áp dụng chiến lược FinOps toàn diện đã giúp NimbusAI chuyển đổi từ mô hình tiêu dùng lãng phí sang mô hình hạ tầng được tối ưu hóa sâu:

| Chỉ số kinh tế FinOps | Baseline (Ban đầu) | Optimized (Sau tối ưu) | Mức cắt giảm / Cải thiện |
|---|---|---|---|
| **Tổng chi phí GPU hàng tháng** | **$27,133 / tháng** | **$14,626 / tháng** | **Tiết kiệm $12,507 (-46.1%)** |
| **Đơn giá Inference (`$/1M-token`)** | **$6.488 / 1M-token** | **$1.126 / 1M-token** | **Giảm 82.6%** |
| **Chi phí Inference hàng ngày** | $48.87 / ngày | $8.48 / ngày | Giảm 82.6% |
| **Chi phí Mua sắm Workloads (Purchasing)** | $25,667 / tháng | $15,627 / tháng | Tiết kiệm 39.1% |
| **Mức độ hoàn thiện Phân bổ chi phí** | Showback (chưa đủ dữ liệu) | **Chargeback Ready (92% Tag Coverage)** | Đạt điều kiện thu phí nội bộ |

> **Nhận xét cốt lõi:** Việc đo lường bằng `$/1M-token` thay vì `$/GPU-giờ` giúp phản ánh chính xác hiệu quả kinh doanh. Giá thuê GPU có thể giữ nguyên hoặc giảm nhẹ, nhưng thông lượng phục vụ trên mỗi USD đầu tư đã tăng gấp gần 6 lần.

---

## 2. Phân tích Chi tiết 4 Đòn bẩy Tiết kiệm (Savings Breakdown)

Tổng mức tiết kiệm **$12,507 / tháng** được đóng góp từ 4 đòn bẩy cụ thể:

| Thứ tự | Đòn bẩy FinOps | Tiết kiệm (USD/tháng) | Tỷ trọng đóng góp | Cơ chế kỹ thuật chính |
|:---:|---|:---:|:---:|---|
| **1** | **Purchasing Strategy (Spot / Reserved)** | **$10,040** | **80.3%** | Chuyển các job training gián đoạn được sang Spot (chiết khấu ~60-70%); cam kết 3-Year Reserved cho các job inference 24/7 (chiết khấu 45%). |
| **2** | **Inference Cost Levers (Cascade / Cache / Batch)** | **$1,212** | **9.7%** | Kết hợp tam giác đòn bẩy: Cascade (chuyển prompt dễ sang model nhỏ rẻ hơn 93%), Prompt Caching (giảm 90% input), Batch API (giảm 50% non-realtime). |
| **3** | **Right-sizing Util-Lies** | **$655** | **5.2%** | Hạ cấp GPU bị "GPU-Util Lie" do nghẽn băng thông bộ nhớ (H100 decode hạ xuống A100; A10G hạ xuống L4) mà không giảm throughput. |
| **4** | **Kill Idle GPUs** | **$600** | **4.8%** | Tắt bỏ các instance GPU bị bỏ quên, không có tiến trình chạy (utilization < 10%). |

### Đòn bẩy nào đóng góp lớn nhất?
- **Purchasing Strategy** đóng góp lớn nhất về mặt **tổng số tiền tuyệt đối ($10,040/tháng, chiếm 80.3% tổng savings)**, vì các cụm huấn luyện H100/A100 ngốn ngân sách nền tảng lớn nhất.
- Tuy nhiên, **Inference Levers** lại có **tỷ lệ phần trăm cắt giảm ấn tượng nhất (82.6%)**, giúp đơn giá phục vụ trực tiếp cho người dùng cuối giảm từ $6.488 xuống chỉ còn $1.126 trên mỗi triệu token.

---

## 3. Bản chất của "GPU-Util Lie" & Tác động Tài chính

### Hiện tượng phát hiện:
Trong quá trình kiểm toán M1, telemetry phát hiện 2 GPU rơi vào trạng thái "GPU-Util Lie" (`util >= 90%` nhưng `MFU < 0.30`):
1. **`gpu-h100-4`**: Báo cáo **GPU-Util = 98.2%**, nhưng **MFU chỉ đạt 0.194 (19.4%)** và MBU đạt 0.207.
2. **`gpu-a10g-1`**: Báo cáo **GPU-Util = 96.9%**, nhưng **MFU chỉ đạt 0.268 (26.8%)** và MBU đạt 0.302.

### Nguyên nhân kỹ thuật phần cứng (Root Cause):
- Lệnh `nvidia-smi` đo `gpu_util_pct` dựa trên tỷ lệ thời gian có ít nhất 1 kernel active trên GPU trong cửa sổ lấy mẫu. Con số này chỉ đo xem "clock có bận hay không", **hoàn toàn không đo lường khối lượng tính toán Tensor Cores thực sự hoàn thành**.
- Trong giai đoạn LLM generation (autoregressive decode), mô hình sinh từng token một. Cường độ tính toán (Arithmetic Intensity) chỉ đạt **~1–2 FLOP/byte**, nằm sâu trong vùng **Memory-Bound** của Roofline Model (trong khi điểm bão hòa ridge point của H100 lên tới **~295 FLOP/byte**).
- Các vi xử lý SM liên tục bị **Memory Stall** để chờ nạp trọng số từ bộ nhớ HBM3 vào SRAM. GPU báo "bận 98%" nhưng thực chất là đang "bận chờ dữ liệu".

### Tác động tài chính:
NimbusAI đang trả tiền trần **$2.50/giờ** cho một chiếc H100 hàng đầu thế giới, nhưng chỉ nhận được sức mạnh tính toán của một chip tầm trung. Việc hạ cấp xuống A100 ($1.79/giờ) hoặc L4 ($0.80/giờ) giúp bảo toàn 100% throughput sinh token mà tiết kiệm ngay **$655/tháng** trên mỗi node.

---

## 4. Báo cáo Chi tiết Các Phần Mở Rộng Đã Thực Hiện (Extensions)

### 4.1. Extension 1 — Nâng cấp Chính sách Mua sắm `recommend_tier()`
- **Vị trí sửa đổi:** [`finops/pricing.py`](finops/pricing.py) và [`missions/m3_purchasing.py`](missions/m3_purchasing.py).
- **Cải tiến logic:**
  - Tích hợp ma trận rủi ro bị thu hồi (Eviction Rate) thực tế theo dòng GPU: H100 (3%), H200 (4%), A100 (5%), L4 (8%), A10G (12%).
  - Bổ sung kiểm tra thời hạn dự án (`job_days`): Nếu job ngắn ngày (< 14 ngày), hệ thống ngăn chặn việc ký cam kết Reserved dài hạn nhằm tránh rủi ro "bẫy cam kết" (financial lock-in).
  - Với các job có nguy cơ ngắt cao (>10%) như dev sandbox trên A10G, cảnh báo rủi ro gián đoạn công việc của kỹ sư.
- **Kết quả đo lường:**
  - 5 job training/batch fault-tolerant (`job-train-llm`, `job-train-embed`, `job-finetune`, `job-batch-eval`, `job-dev-sandbox`) được phân bổ sang Spot, tiết kiệm từ 35% đến 58% chi phí từng job.
  - 3 job inference chạy 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`) có duty cycle = 100% (vượt xa điểm hòa vốn 55%) được cam kết 3-Year Reserved, tiết kiệm đều đặn 40–45% mỗi tháng.

### 4.2. Extension 3 — Kinh tế học Prompt Caching (`cache_is_worth_it`)
- **Vị trí sửa đổi:** [`finops/pricing.py`](finops/pricing.py) và [`missions/m2_inference_levers.py`](missions/m2_inference_levers.py).
- **Cơ chế kinh tế:**
  - Nhà cung cấp tính phí lưu trữ/ghi cache ban đầu ($C_{write} \approx 1.25 \times P_{in}$).
  - Mỗi lượt đọc cache tiết kiệm: $S_{read} = P_{in} \times (1 - 0.10) = 0.90 \times P_{in}$.
  - **Điểm hòa vốn (Break-even reads):**
    $$N_{break-even} = \frac{C_{write}}{S_{read}} = \frac{1.25 \times P_{in}}{0.90 \times P_{in}} \approx 1.39 \text{ lần đọc}$$
- **Kết quả đo lường trên dữ liệu NimbusAI:**
  - Trong `token_usage.csv`, toàn bộ 2,400 request đều có thành phần cached prompt prefix (chiếm 31.9% tổng input tokens).
  - Tần suất đọc lại trung bình đạt > 5 lượt/prefix, vượt xa ngưỡng hòa vốn 1.39 lần.
  - Do đó, `cache_is_worth_it` trả về `True`, xác nhận việc bật prompt cache là **sinh lời tuyệt đối**, giúp chi phí token giảm mạnh.

### 4.3. Extension 4 — Phân tích Ngân sách Reasoning & Tác động Năng lượng
- **Vị trí sửa đổi:** [`missions/m2_inference_levers.py`](missions/m2_inference_levers.py) và [`finops/report.py`](finops/report.py).
- **Phát hiện chấn động:**
  - **Lưu lượng traffic:** Reasoning queries (`is_reasoning=1`) chỉ chiếm **201 / 2,400 request (8.4%)** và 16.5% tổng số token.
  - **Chi phí tiền:** Chiếm $1.40 / $8.48/ngày (16.5% chi phí inference sau tối ưu).
  - **Tiêu thụ điện năng:** Tiêu thụ tới **29,787.7 Wh / 31,675.3 Wh (chiếm 94.0% tổng năng lượng toàn hệ thống inference!)**.
  - **Tỷ lệ chênh lệch:** Một request reasoning tiêu thụ năng lượng **gấp 15.8 lần** so với toàn bộ 91.6% lượng query tiêu chuẩn gộp lại!
- **Đề xuất Routing Policy:**
  - Không mở reasoning tự do cho toàn bộ prompt.
  - Thiết lập bộ phân loại độ phức tạp (Task Complexity Gating): Chỉ định tuyến sang reasoning model khi độ phức tạp vượt ngưỡng hoặc bài toán đòi hỏi suy luận nhiều bước (math, code synthesis). Khống chế traffic reasoning dưới 10% để ngăn chặn bùng nổ chi phí điện và dấu chân carbon.

---

## 5. Khuyến nghị Kế hoạch Hành động cho Lãnh đạo NimbusAI

Nếu đảm nhận vị trí Lead FinOps tại NimbusAI, tôi đề xuất lộ trình hành động 4 bước có thứ tự ưu tiên ROI rõ ràng:

1. **Hành động 1 (P0 - Ngay lập tức / Ngày 1): Tắt GPU Idle & Cài đặt Auto-Termination**
   - *Hành động:* Thu hồi ngay node `gpu-h100-5` bị bỏ trống 8h/ngày và cài đặt daemon script tự động shutdown bất kỳ GPU nào có utilization < 5% trong 30 phút.
   - *ROI:* Tiết kiệm ngay **$600 / tháng** với zero rủi ro nghiệp vụ.
2. **Hành động 2 (P1 - Tuần 1): Tái cấu trúc Mua sắm (Spot + Reserved)**
   - *Hành động:* Cấu hình S3/GCS checkpoint định kỳ cho các pipeline huấn luyện `job-train-llm`, `job-train-embed` để chuyển toàn bộ sang Spot. Ký cam kết 3-Year Reserved cho các cụm inference ổn định 24/7.
   - *ROI:* Cắt giảm **$10,040 / tháng** (đóng góp tới 80.3% mục tiêu tài chính).
3. **Hành động 3 (P2 - Tuần 2): Triển khai Gateway Cascade Routing & Prompt Caching**
   - *Hành động:* Triển khai Proxy Gateway (như LiteLLM) để tự động phân luồng query đơn giản về model nhỏ, bật tính năng Prompt Caching cho system prompts và định tuyến tác vụ phi thời gian thực sang Batch API.
   - *ROI:* Giảm đơn giá inference từ **$6.488 xuống $1.126 / 1M-token** (giảm 82.6%).
4. **Hành động 4 (P3 - Tháng 1): Hạ cấp Right-sizing & Di dời sang Green Region**
   - *Hành động:* Thay thế các node H100 chạy decode LLM bằng A100/L4, tiết kiệm thêm **$655 / tháng**. Đồng thời chuyển các cụm training sang vùng **`europe-north1`** (Na Uy - thủy điện, 30 gCO2/kWh) để cắt giảm 92% lượng phát thải carbon so với `us-east-1`.

---

## 6. Danh mục Hồ sơ Nghiệm thu Đính kèm

- Báo cáo định dạng Markdown: [`outputs/report.md`](outputs/report.md)
- Biểu đồ Waterfall các đòn bẩy tiết kiệm: [`outputs/savings.png`](outputs/savings.png)
- Bảng kê chi phí chuẩn quốc tế FOCUS: [`outputs/focus_export.csv`](outputs/focus_export.csv)
- Bộ kiểm tra tự động: `verify.py` đạt **11/11 PASS**, `pytest` đạt **17/17 PASS**.
