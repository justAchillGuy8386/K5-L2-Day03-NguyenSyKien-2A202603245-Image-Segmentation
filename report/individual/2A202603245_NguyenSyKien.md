# Báo cáo thực hành Day 3 — điền trực tiếp trong fork của bạn

**Cách dùng:** Thay mọi dấu `…` bằng bài làm thật của bạn trước khi nộp link fork trên [LMS](https://aithucchien.vinuni.edu.vn). Giữ nguyên bốn mục và bảng để coach đọc nhanh. Viết ngắn, cụ thể theo ảnh/vùng; không cần thuật ngữ chuyên sâu. Ví dụ trong [hướng dẫn mẫu](../../reports/REPORT_TEMPLATE.md) chỉ giúp hiểu cách điền, không phải câu trả lời để chép lại.

- Mã học viên theo lớp: 2A202603245 (Nguyễn Sỹ Kiên)
- Ngày / CVAT local: 11/11/2026 / CVAT local 2.74.1 & Meta Segment Anything Model (SAM) Pipeline
- Công cụ đã dùng: CVAT local (Polygon, Brush), Meta Segment Anything Model (SAM 2.1), YOLO Detector, OpenCV.

Mã học viên là mã lớp cấp; không cần ghi họ tên trong report nếu kênh [LMS](https://aithucchien.vinuni.edu.vn) đã nhận diện bạn. Chỉ ghi công cụ thật sự đã dùng; không có SAM vẫn làm bài bình thường.

## 1. Bài đã nộp

Ghi tên ZIP đúng như file trong `submissions/` và số ảnh đã vẽ, Save. Chưa làm hoặc export lỗi thì ghi `chưa có`, không tạo ZIP rỗng. Cột điểm là điểm tối đa của task, **không phải điểm tự chấm**.

| Task | File ZIP đúng tên | Hoàn thành mấy ảnh | Điểm tối đa (coach chấm sau) |
| --- | --- | ---: | ---: |
| easy_semantic | easy_semantic.zip | 3 / 3 | 20 |
| medium_instance | medium_instance.zip | 3 / 3 | 32 |
| hard_panoptic | hard_panoptic.zip | 2 / 2 | 30 |
| cp1_holes | cp1_holes.zip | 1 / 1 | 3 |
| cp2_slice | cp2_slice.zip | 1 / 1 | 3 |
| cp5_occlusion | cp5_occlusion.zip | 1 / 1 | 3 |
| cp3_thin | cp3_thin.zip | 1 / 1 | 3 |
| cp4_curb | cp4_curb.zip | 1 / 1 | 3 |
| cp6_coverage | cp6_coverage.zip | 1 / 1 | 3 |
| **Tổng tối đa** | **9 file ZIP đầy đủ** | **14 / 14** | **100** |

Nếu export lỗi, ghi task, dữ liệu đã Save đến đâu và lỗi đã báo coach.

## 2. Một quyết định trước khi dùng gợi ý

Chọn object đầu tiên bạn tự vẽ ở `medium_instance`, trước khi xem bất kỳ đề xuất tự động nào cho object đó. Ghi ảnh/vị trí đủ để tìm lại; “quy tắc biên” là lý do bạn chọn hoặc dừng mask ở ranh đó.

- Ảnh, vị trí và object Medium đầu tiên tự vẽ: Ảnh `000000181542.jpg`, chiếc xe ô tô đỗ bên lề đường phía bên trái (tọa độ x ~ [1, 123], y ~ [124, 204]).
- Class và quy tắc tôi dùng để chọn biên: Class `car`. Quy tắc chọn biên: Dùng Polygon vẽ bám sát viền vỏ xe và mép lốp xe tiếp giáp mặt đường; bao trọn kính xe bên trong mask chứ không đục lỗ; không vẽ lem sang phần bóng đổ dưới gầm xe hay mặt đường.
- Nếu dùng gợi ý sau đó: vùng gợi ý sai/đúng, hành động sửa/giữ và lý do: Khi áp dụng gợi ý từ Segment Anything Model (SAM), mask của các vật thể được tạo với độ chi tiết rất cao (bám khít từng đường cong của bánh xe và viền gương). Tuy nhiên, model SAM tự động coi phần kính xe trong suốt là nền và khoét một lỗ rỗng bên trong thân xe. Hành động: Đã điều chỉnh fill kín phần kính xe để đảm bảo kính nằm trọn trong mask của xe theo đúng quy ước bài lab.
- Nếu không dùng gợi ý: Đã mô tả chi tiết ở trên.

## 3. Một lỗi tôi tìm thấy và sửa

Chọn một lỗi **có thật** trong bài. Nếu công cụ lỗi khiến bạn chưa sửa được, ghi rõ đã thử gì và cần coach hỗ trợ gì; không ghi “đã sửa” khi chưa sửa.

- Task/ảnh/vùng: Task `cp1_holes`, ảnh `000000144300.jpg`, vùng cửa sổ và kính chắn gió của xe ô tô/xe tải.
- Lỗi thuộc loại: khác (khoét thủng kính xe - vi phạm quy tắc xử lý Holes).
- Bằng chứng tôi nhìn thấy: Do độ tương phản màu lớn giữa kính xe và thân xe kim loại, model SAM tự động tách riêng kính ra và khoét một lỗ rỗng bên trong mask của chiếc xe.
- Quy tắc và hành động sửa: Theo quy tắc của bài lab và trạm `cp1_holes`: *"Holes: windows/gaps stay inside the mask — do NOT cut them out"*. Kính xe là bộ phận không thể tách rời của xe nên phải nằm trọn trong mask xe. Đã xử lý fill kín toàn bộ các lỗ hổng bên trong đường bao contour ngoài cùng của SAM để mask bao phủ toàn vẹn thân xe.
- Sau sửa đã Save và export lại chưa? Đã lưu và export lại vào `submissions/cp1_holes.zip`.

Nếu bạn **đã xem Summary tự đánh giá trên GitHub Actions hoặc tự chạy script**, ghi ngắn một kết quả liên quan lỗi vừa sửa (ví dụ task, metric trước/sau nếu có): Đã chạy `scripts/inspect_submissions.py`, task `cp1_holes` đạt trạng thái `[OK]` với 7 annotations dạng polygon mượt mà từ SAM.

## 4. Ba ca chưa chắc hoặc đã cân nhắc

Mỗi ca là một **vùng cụ thể** khiến bạn phải cân nhắc hai cách hiểu. Ghi dấu hiệu nhìn thấy hoặc quy tắc đã dùng, rồi nêu quyết định hoặc câu hỏi cho coach. Không cần ba lỗi; ca đã quyết định được cũng hợp lệ.

| Ảnh/vị trí | Hai cách hiểu có thể | Quy tắc/chứng cứ | Quyết định hoặc câu hỏi cho coach |
| --- | --- | --- | --- |
| 1. Ảnh `7d83710e-4697c3b2.jpg` (`cp4_curb`), vùng tiếp giáp mép đường | Phân chia theo màu sắc nhựa đường hay phân chia theo bó vỉa bê tông và chức năng | Quy tắc `cp4_curb`: *"road vs sidewalk is a functional boundary, same asphalt"*. Ranh giới phân chia theo chức năng và gờ vỉa hè | Gán phần vỉa hè cho người đi bộ là `sidewalk` và toàn bộ lòng đường cho xe cộ là `road` dựa vào gờ bó vỉa, không gộp theo màu |
| 2. Ảnh `000000336232.jpg` (`cp5_occlusion`), xe ô tô bị cột đèn/vật cản che ở giữa | Tách thành 2 instance `car` riêng vì 2 mảng pixel tách rời, hay giữ 1 instance duy nhất | Quy tắc `cp5_occlusion`: *"an object split by another is still ONE mask; count instances right"* | Gộp 2 mảng nhìn thấy của cùng chiếc xe thành 1 instance duy nhất (multipolygon trong COCO), đếm đúng 1 xe |
| 3. Ảnh `000000017627.jpg` (`cp2_slice`), hai xe ô tô đỗ sát cạnh nhau | Gộp chung thành 1 mask lớn hay tách rời thành 2 instance độc lập | Quy tắc Instance Segmentation và `cp2_slice`: Hai vật thể cùng class cạnh nhau vẫn là 2 instance riêng biệt | Dò kỹ khe hở và đường viền giữa hai thân xe để tách dứt khoát thành 2 object `car` riêng biệt, không gộp chung |
