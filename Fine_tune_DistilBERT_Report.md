# Báo cáo: Fine-tune DistilBERT cho bài toán phát hiện SMS Spam

## 1. Giới thiệu

### 1.1. Mục tiêu
Nghiên cứu và thực nghiệm phương pháp **Fine-tuning** mô hình ngôn ngữ pre-trained **DistilBERT** để phân loại tin nhắn SMS thành hai lớp: **ham** (tin nhắn bình thường) và **spam** (tin nhắn rác).

### 1.2. Mô hình sử dụng
**DistilBERT** (`distilbert-base-uncased`) là phiên bản rút gọn của BERT, được phát triển bởi Hugging Face với các đặc điểm:
- Nhỏ hơn BERT 40% (66 triệu tham số so với 110 triệu)
- Nhanh hơn BERT 60% khi inference
- Giữ lại 97% hiệu năng của BERT
- Được huấn luyện bằng kỹ thuật **Knowledge Distillation** từ BERT gốc

### 1.3. Công cụ và thư viện
- **Google Colab**: Môi trường chạy notebook với GPU miễn phí
- **Hugging Face Transformers**: Thư viện cung cấp mô hình pre-trained và API fine-tuning
- **Hugging Face Datasets**: Quản lý dữ liệu dạng dataset
- **Hugging Face Evaluate**: Tính toán các metric đánh giá
- **scikit-learn**: Chia dữ liệu train/test, confusion matrix, classification report
- **pandas**: Xử lý dữ liệu dạng bảng
- **matplotlib, seaborn**: Trực quan hóa kết quả
- **LIME**: Giải thích dự đoán của mô hình (chuẩn bị cho phân tích sau)

---

## 2. Dữ liệu

### 2.1. Nguồn dữ liệu
Sử dụng bộ dữ liệu **SMS Spam Collection** từ Kaggle — một bộ dữ liệu phổ biến trong nghiên cứu phát hiện spam. File CSV gốc chứa 5,572 tin nhắn SMS đã được gán nhãn thủ công.

### 2.2. Cấu trúc dữ liệu gốc
File CSV gốc có 5 cột:
| Cột | Mô tả |
|-----|--------|
| `v1` | Nhãn phân loại: `ham` hoặc `spam` |
| `v2` | Nội dung tin nhắn SMS |
| `Unnamed: 2, 3, 4` | Các cột thừa, chứa toàn giá trị NaN |

### 2.3. Tiền xử lý dữ liệu
Các bước tiền xử lý được thực hiện:

1. **Loại bỏ cột thừa**: Xóa 3 cột `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4` (chứa toàn NaN)
2. **Đổi tên cột**: `v1` → `label`, `v2` → `text` cho dễ đọc
3. **Loại bỏ giá trị NaN**: Xóa các dòng có giá trị thiếu
4. **Loại bỏ duplicate**: Xóa các tin nhắn trùng lặp
   - Trước: 5,572 mẫu
   - Sau: **5,169 mẫu** (loại bỏ 403 mẫu trùng)
5. **Mã hóa nhãn**: Chuyển nhãn dạng text sang số
   - `ham` → 0
   - `spam` → 1

### 2.4. Phân bố dữ liệu sau tiền xử lý

| Nhãn | Số lượng | Tỉ lệ |
|------|----------|--------|
| Ham (tin bình thường) | 4,516 | 87.37% |
| Spam (tin rác) | 653 | 12.63% |
| **Tổng** | **5,169** | **100%** |

Dữ liệu có sự **mất cân bằng** giữa hai lớp — lớp spam chỉ chiếm khoảng 1/8 tổng số mẫu. Đây là đặc điểm phổ biến trong bài toán phát hiện spam thực tế.

### 2.5. Đặc điểm nội dung tin nhắn

**Ví dụ tin nhắn ham:**
- "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
- "Ok lar... Joking wif u oni..."
- "U dun say so early hor... U c already then say..."

**Ví dụ tin nhắn spam:**
- "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005..."
- "FreeMsg Hey there darling it's been 3 week's now and no word back!..."
- "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!..."

### 2.6. Phân bố độ dài tin nhắn
- Độ dài được đo bằng số từ trong mỗi tin nhắn
- Phần lớn tin nhắn có độ dài ngắn, phù hợp với đặc trưng của SMS
- Percentile thứ 95 được đánh dấu trên biểu đồ để xác định ngưỡng độ dài phù hợp cho tokenizer

---

## 3. Phương pháp

### 3.1. Khái niệm Fine-tuning
Fine-tuning là kỹ thuật **chuyển giao học** (Transfer Learning), trong đó:
- Lấy một mô hình đã được **pre-trained** trên lượng lớn dữ liệu tổng quát (DistilBERT được pre-trained trên Wikipedia + BookCorpus)
- **Tinh chỉnh** (fine-tune) toàn bộ tham số của mô hình trên dữ liệu cụ thể của bài toán (SMS Spam)
- Mô hình tận dụng được kiến thức ngôn ngữ đã học từ trước, chỉ cần điều chỉnh cho phù hợp với task mới

### 3.2. Kiến trúc mô hình
Mô hình `DistilBertForSequenceClassification` bao gồm:
- **Backbone DistilBERT** (đã pre-trained): Biến đổi text thành vector biểu diễn ngữ nghĩa
- **Pre-classifier layer**: Lớp fully-connected trung gian
- **Classifier head** (khởi tạo random): Lớp phân loại 2 đầu ra (ham/spam)

Tổng số tham số: **66,955,010** (tất cả đều được fine-tune)

### 3.3. Tokenization
Sử dụng tokenizer của DistilBERT với các thiết lập:
- **max_length = 128**: Độ dài tối đa của chuỗi token (phù hợp với SMS ngắn)
- **padding = "max_length"**: Đệm các chuỗi ngắn hơn 128 token về cùng độ dài
- **truncation = True**: Cắt bỏ phần vượt quá 128 token

Ví dụ tokenize:
```
Input:  "Free entry to win iPhone now!"
Tokens: ['[CLS]', 'free', 'entry', 'to', 'win', 'iphone', 'now', '!', '[SEP]']
```

### 3.4. Chia dữ liệu
- **Train set**: 4,135 mẫu (80%) — 12.6% spam
- **Test set**: 1,034 mẫu (20%) — 12.7% spam
- Sử dụng `stratify` để đảm bảo tỉ lệ spam/ham nhất quán giữa hai tập
- `random_state=42` để đảm bảo tính tái lập

### 3.5. Cấu hình huấn luyện

| Hyperparameter | Giá trị | Giải thích |
|----------------|---------|------------|
| Số epochs | 3 | Số lần duyệt qua toàn bộ tập train |
| Batch size (train) | 16 | Số mẫu trong mỗi bước cập nhật |
| Batch size (eval) | 32 | Số mẫu khi đánh giá |
| Learning rate | 2e-5 | Tốc độ học — giá trị nhỏ phù hợp cho fine-tuning |
| Weight decay | 0.01 | Regularization để tránh overfitting |
| Eval strategy | Mỗi epoch | Đánh giá trên test set sau mỗi epoch |
| Metric chọn model tốt nhất | F1-score | Ưu tiên cân bằng precision và recall |
| Save total limit | 1 | Chỉ giữ 1 checkpoint tốt nhất |

### 3.6. Metric đánh giá
- **Accuracy**: Tỉ lệ dự đoán đúng trên toàn bộ dữ liệu
- **F1-score**: Trung bình điều hòa của Precision và Recall — metric chính cho bài toán mất cân bằng
- **Precision**: Trong các mẫu được dự đoán là spam, bao nhiêu % thực sự là spam
- **Recall**: Trong các mẫu thực sự là spam, bao nhiêu % được phát hiện đúng
- **Confusion Matrix**: Ma trận nhầm lẫn thể hiện chi tiết các loại lỗi

---

## 4. Kết quả

### 4.1. Quá trình huấn luyện
- Tổng số bước huấn luyện: **777 steps**
- Training loss trung bình: **0.0463** — cho thấy mô hình hội tụ tốt
- Thời gian huấn luyện: **~183 giây** (~3 phút)
- Tốc độ: 67.6 mẫu/giây

### 4.2. Kết quả đánh giá trên Test Set

| Metric | Giá trị |
|--------|---------|
| **Eval Loss** | 0.0345 |
| **Accuracy** | 99.03% |
| **F1-score (spam)** | 96.12% |
| **Precision (spam)** | 97.64% |
| **Recall (spam)** | 94.66% |

### 4.3. Classification Report chi tiết

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Ham | 0.9923 | 0.9967 | 0.9945 | 903 |
| Spam | 0.9764 | 0.9466 | 0.9612 | 131 |
| **Accuracy** | | | **0.9903** | **1,034** |
| **Macro avg** | 0.9843 | 0.9716 | 0.9779 | 1,034 |
| **Weighted avg** | 0.9903 | 0.9903 | 0.9903 | 1,034 |

### 4.4. Confusion Matrix

|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actual Ham** | 900 | 3 |
| **Actual Spam** | 7 | 124 |

- **True Positive (spam đúng)**: 124 mẫu
- **True Negative (ham đúng)**: 900 mẫu
- **False Positive (ham bị nhận nhầm thành spam)**: 3 mẫu
- **False Negative (spam bị bỏ sót)**: 7 mẫu

### 4.5. Phân tích các mẫu dự đoán sai
Tổng số mẫu sai: **10/1,034** (tỉ lệ lỗi ~0.97%)

**False Positive — 3 mẫu ham bị nhận nhầm thành spam:**
Các tin nhắn ham chứa ngôn ngữ dễ nhầm lẫn với spam:
- Tin nhắn chứa thông số kỹ thuật sản phẩm ("10.1mega pixels, 3optical and 5digital dooms")
- Tin nhắn đề cập đến giá tiền ("Your bill at 3 is £33.65")
- Tin nhắn có phong cách viết giống quảng cáo

**False Negative — 7 mẫu spam bị bỏ sót:**
Các tin nhắn spam có đặc điểm ranh giới, không chứa từ khóa spam điển hình:
- Spam dạng nội dung người lớn, không chứa các từ "FREE", "WINNER"
- Spam dạng câu đố/trò chơi trá hình
- Spam dạng thông báo dịch vụ trông giống tin nhắn hệ thống
- Spam dạng tin tức/sự kiện

### 4.6. Kết quả test thực tế
Thử nghiệm với 7 tin nhắn mới (không có trong tập dữ liệu):

| Tin nhắn | Dự đoán | Confidence |
|----------|---------|------------|
| "Hey, are we still meeting at 3pm tomorrow?" | Ham | 99.87% |
| "WINNER!! You've won $1000! Click here to claim now!" | Spam | 98.89% |
| "Mom, I'll be home late tonight" | Ham | 99.93% |
| "URGENT! Your account will be suspended. Reply YES to verify." | Spam | 91.08% |
| "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005" | Spam | 99.71% |
| "I'll call you when I get home, love you" | Ham | 99.93% |
| "Congratulations! You have been selected for a $500 gift card. Text WIN to 12345" | Spam | 99.84% |

Tất cả 7 mẫu đều được phân loại **chính xác** với độ tin cậy cao.

---

## 5. Nhận xét và kết luận

### 5.1. Về phương pháp Fine-tuning
- Fine-tuning DistilBERT cho thấy hiệu quả rõ rệt trên bài toán phân loại SMS spam
- Chỉ cần **3 epochs** (~3 phút training) trên **~4,100 mẫu** đã đạt F1 = 96.12%
- Điều này chứng minh sức mạnh của Transfer Learning: mô hình pre-trained đã có sẵn kiến thức ngôn ngữ, chỉ cần tinh chỉnh nhẹ là phù hợp với bài toán cụ thể

### 5.2. Về kết quả
- Accuracy 99.03% và F1-score 96.12% là kết quả tốt cho bài toán phân loại nhị phân
- Precision cao (97.64%) giúp hạn chế tình trạng tin nhắn bình thường bị đánh nhầm thành spam
- Recall đạt 94.66% nghĩa là mô hình phát hiện được phần lớn tin nhắn spam
- Các mẫu bị phân loại sai đều nằm ở vùng ranh giới, khó phân biệt ngay cả với con người

### 5.3. Về pipeline thực nghiệm
Pipeline thực nghiệm được xây dựng hoàn chỉnh, bao gồm đầy đủ các bước:
1. Load và khám phá dữ liệu
2. Tiền xử lý và trực quan hóa
3. Tokenize với DistilBERT tokenizer
4. Fine-tune với Hugging Face Trainer API
5. Đánh giá định lượng (accuracy, F1, precision, recall)
6. Đánh giá định tính (confusion matrix, phân tích lỗi, test thực tế)
7. Lưu mô hình để tái sử dụng

Mô hình đã được lưu để phục vụ cho việc phân tích giải thích bằng LIME trong các bước tiếp theo.
