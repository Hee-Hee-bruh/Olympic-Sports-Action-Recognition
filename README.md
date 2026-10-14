# Olympic Sports Action Recognition

Triển khai theo Introduction và phương pháp trong bản
`output/olympic_sports_class_report.tex` của thư mục dự án phía trên
(bản báo cáo mới nhất). Pipeline Python độc lập với notebook cũ.

Hai câu hỏi nghiên cứu:

- **RQ1:** Thêm motion vào ResNet có cải thiện phân loại không? So sánh D với B.
- **RQ2:** ResNet pretrained có tốt hơn HOG, và chi phí trích đặc trưng khác nhau thế nào? So sánh B với A.

| ID | Đặc trưng video | Số chiều mặc định | Classifier |
|---|---|---:|---|
| A | HOG trung bình, L2 | 6.084 | Linear SVM, one-vs-rest |
| B | ResNet-18 ImageNet frozen, trung bình, L2 | 512 | Linear SVM, one-vs-rest |
| C | Farnebäck flow, histogram hướng có trọng số magnitude, trung bình, L2 | 32 | Linear SVM, one-vs-rest |
| D | Ghép B và C sau khi chuẩn hóa từng block | 544 | Linear SVM, one-vs-rest |

## Chạy project

Cần **Python 3.11 trở lên**. Trong máy hiện tại, môi trường `.venv` đã được cài
và có thể dùng ngay. Với máy mới, tạo môi trường bằng Python phù hợp:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
```

Từ thư mục repo:

```bash
source .venv/bin/activate

# Tải release Stanford cùng train/test split chính thức; khoảng 5,2 GB archive.
# Archive và các .seq giải nén được giữ lại trong data/ và không đưa lên Git.
python scripts/download_dataset.py --workers 4

# Tạo danh sách clip từ các file train/*.txt và test/*.txt.
olympic prepare

# Kiểm tra file, đọc frame và tạo cache cho cả ba loại đặc trưng.
olympic extract

# Chọn C trên validation. Lệnh này chưa xuất kết quả test.
MPLCONFIGDIR=.cache/matplotlib olympic experiment

# Sau khi chốt preprocessing/sampling: fit toàn bộ train và đánh giá test.
MPLCONFIGDIR=.cache/matplotlib olympic experiment --evaluate-test

# Nhận dạng một clip mới bằng model đã lưu.
olympic predict /path/to/clip.seq --model runs/baseline/resnet_flow.joblib
```

Trọng số ResNet-18 chính thức được tải lần đầu về `.cache/features/torch/`.
Pipeline cần Internet khi tải dataset và trọng số, sau đó chạy được offline.
Các quyết định của SVM là **decision scores**, không phải xác suất.

## Chuẩn bị và xử lý dữ liệu

Nguồn: [Stanford Olympic Sports, release 2010.09.07](http://vision.stanford.edu/Datasets/OlympicSports/).
Trang hiện tại truy cập được bằng HTTP; downloader dùng đúng URL archive được
trang này công bố, giữ SHA-256 và URL trong `data/download_provenance.json`.
Split đã kiểm tra: **649 train, 134 test**, 16 lớp.
Không tạo một test split ngẫu nhiên mới.

Reader đọc trực tiếp Norpix `.seq` 8-bit raw/JPEG/PNG, với frame padding 8 hoặc
16 byte, dựa trên [đặc tả reader của Piotr's Toolbox](https://github.com/pdollar/toolbox/blob/master/videos/seqReaderPlugin.m).
Không cần chuyển sang AVI. `.mp4`, `.avi`, `.mov`, `.mkv` cũng được hỗ trợ qua
OpenCV nếu bạn dùng dữ liệu đã chuyển đổi.

Manifest có `clip_id,path,label,split`; đường dẫn tương đối được hiểu theo vị trí
file manifest. `prepare` ánh xạ tên lớp trong split chính thức, bao gồm
`platform 10m` → `diving_platform_10m` và `springboard 3m` → `diving_springboard_3m`.
Clip có cùng ID hoặc cùng đường dẫn ở nhiều dòng sẽ bị từ chối.
SHA-256 giống nhau được ghi nhận là trùng nội dung và cả hai clip bị loại khỏi
tất cả cấu hình. File thiếu, lỗi decode hoặc clip dưới 2 frame được ghi vào audit.
Nếu một lớp biến mất khỏi train/validation/test, đánh giá sẽ dừng với lỗi rõ ràng.

Với clip T frame, chọn tối đa K=10 chỉ số duy nhất trên `[0, T-2]`, theo khoảng
cách đều. Appearance dùng frame tại mỗi chỉ số; flow dùng frame đó và frame
gốc **liền sau**, không tính giữa hai frame cách xa trong danh sách sample.
Tất cả ảnh resize trực tiếp về 224×224; HOG/flow dùng grayscale, ResNet dùng RGB
và mean/std ImageNet. [Trọng số torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18)
được cố định ở `IMAGENET1K_V1`; resize trực tiếp theo đề cương thay vì dùng
resize 256 + center crop mặc định của torchvision. Model ở `eval()`, không cập
nhật weights hoặc batch-normalisation, không augmentation.

## Protocol và đầu ra

`configs/baseline.json` lưu toàn bộ thiết lập. Validation là 20% **chỉ từ train**,
stratified với seed 42. Mỗi cấu hình chọn C trong `{0.01, 0.1, 1, 10}` bằng mAP
validation; nếu bằng nhau, chọn C nhỏ nhất. Sau khi chọn xong **cả bốn cấu hình**,
fit lại trên toàn bộ official train rồi mới tính metrics trên official test.

AP tính one-vs-rest bằng decision scores, mAP là trung bình AP của đủ 16 lớp.
Accuracy lấy nhãn có score cao nhất; macro-F1 và confusion matrix dùng cùng
thứ tự lớp. Các cấu hình dùng cùng giao các clip đọc được.

| File | Nội dung |
|---|---|
| `runs/features_k10/audit.csv` | Metadata, sample indices, SHA-256, decode status, exclusion và cache hit |
| `runs/features_k10/features.npz` | HOG, ResNet, flow, ID, label, official split và thời gian từng clip |
| `runs/features_k10/extraction.json` | Config, versions, manifest và định nghĩa timing |
| `runs/baseline/partitions.csv` | ID chính xác của fit, validation, test |
| `runs/baseline/validation.json` | Kết quả grid C và C được chọn |
| `runs/baseline/protocol.json` | Settings, hardware, versions, class counts và feature hash |
| `runs/baseline/summary.csv` | mAP, accuracy, macro-F1, extraction/train/inference time cho A–D |
| `runs/baseline/per_class_ap.csv` | AP và số mẫu test của từng lớp |
| `runs/baseline/predictions.csv` | Nhãn và decision scores cho từng clip, từng cấu hình |
| `runs/baseline/*_confusion.csv/png` | Confusion matrix chuẩn hóa theo hàng |
| `runs/baseline/*_metrics.json` | Metrics và cả confusion counts nguyên gốc |
| `runs/baseline/*.joblib` | Classifier, method, class order và preprocessing config |
| `runs/baseline/research_questions.json` | D−B, B−A và majority-class accuracy baseline |
| `runs/baseline/error_review.csv` | Tối đa 10 lỗi ưu tiên đa dạng lớp, để ghi quan sát thủ công |

Chỉ nạp `.joblib` do bạn hoặc pipeline này tạo. Chi phí trích đặc trưng gồm
resize, normalization, descriptor và pooling; decode và khởi tạo/warmup ResNet
được tách riêng. Cache giữ thời gian cold run gốc để không đánh đồng đọc cache
với tốc độ trích đặc trưng. D tái sử dụng B/C; extraction cost của D là tổng
B+C, không bao gồm copy ghép vector. `estimated_full_seconds_per_clip` là
ước lượng cộng decode + extraction + cached SVM inference, không phải phép đo
latency end-to-end riêng. Thời gian inference đo theo batch test.

Reader SEQ kiểm tra các record của toàn clip và decode các frame được sample;
không khẳng định đã decode mọi frame. Video thông thường được đọc hai lượt để
đếm frame thực tế và lấy sample; timing decode giữa hai kiểu format có thể khác.
CPU là mặc định để benchmark thống nhất. Có thể đổi `device` sang `mps` hoặc
`cuda` trên phần cứng phù hợp; kết quả giữa thiết bị không được coi là cùng benchmark.

## Ablation K=5 trên validation (tùy chọn)

Tạo bản sao config, đổi `samples` thành 5 và dùng output riêng:

```bash
olympic extract --config configs/k5.json --output runs/features_k5
olympic experiment --config configs/k5.json --features runs/features_k5/features.npz --output runs/validation_k5
```

So sánh B/D với K=10 trên validation để chốt K trước khi đánh giá test.
Không dùng test để quyết định K. Việc đổi config hoặc phiên bản thư viện tạo
cache key khác; vector cũ không được tái sử dụng với preprocessing khác.

## Kiểm thử

```bash
MPLCONFIGDIR=.cache/matplotlib python -m pytest -q
```

Tests kiểm tra SEQ raw/JPEG, padding, truncation, clip ngắn, sampling liền kề,
histogram hướng/magnitude, chuẩn hóa zero-motion, chống trùng manifest,
train-only model selection, thứ tự score và xuất metrics/model/artifacts.
Kiểm thử tổng hợp phục vụ kiểm chứng code; số liệu nghiên cứu phải chạy từ
dataset Stanford thật. Nhận xét lỗi về camera motion, background và pose cần
được xác nhận bằng quan sát clip, không suy ra chỉ từ bảng confusion.
