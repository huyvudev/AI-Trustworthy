# AI-Trustworthy: Fine-tune DistilBERT + LIME cho phân loại SMS Spam

Project môn **Trustworthy AI**: huấn luyện mô hình DistilBERT để phân loại tin nhắn SMS là **ham** (bình thường) hay **spam**, sau đó dùng **LIME** để giải thích **TẠI SAO** model đưa ra mỗi dự đoán — biến model "hộp đen" thành model có thể hiểu được.

---

## 1. Mục tiêu

| Mục tiêu | Mô tả |
|---|---|
| Bài toán | Binary text classification: SMS → `ham` hoặc `spam` |
| Mô hình | DistilBERT (`distilbert-base-uncased`) fine-tune trên SMS Spam Collection |
| Giải thích | Dùng LIME (Local Interpretable Model-agnostic Explanations) để xác định những từ nào đẩy dự đoán về phía `spam` hoặc `ham` |
| Trọng tâm "trustworthy" | Phân tích các trường hợp model dự đoán **SAI** (False Positives & False Negatives) để khám phá điểm yếu của model — tinh thần "Husky vs Wolf" của paper LIME gốc (Ribeiro et al., 2016) |

---

## 2. Cấu trúc thư mục

```
AI-Trustworthy/
├── Fine_tune_DistilBERT_with_LIME.ipynb   # Notebook chính (58 cells)
├── dataset/
│   └── spam.csv                           # SMS Spam Collection (5,572 mẫu, Kaggle)
├── lime_explanations/                     # Output HTML từ LIME
│   ├── correct_ham_1.html ... correct_ham_3.html      # 3 ham dự đoán đúng
│   ├── correct_spam_1.html ... correct_spam_3.html    # 3 spam dự đoán đúng
│   ├── false_positive_1.html ... false_positive_5.html # ham → spam (sai)
│   ├── false_negative_1.html ... false_negative_5.html # spam → ham (sai)
│   └── top_words_analysis.png             # Biểu đồ top từ đẩy về spam/ham
├── .gitattributes
└── README.md                              # File này
```

---

## 3. Dataset

**File:** `dataset/spam.csv` — nguồn từ Kaggle (SMS Spam Collection).

- **Định dạng gốc** có 5 cột: `v1`, `v2`, và 3 cột rỗng `Unnamed: 2/3/4`.
- Chỉ giữ 2 cột:
  - `v1` → `label` (`ham` = 0, `spam` = 1)
  - `v2` → `text` (nội dung SMS)
- **Encoding:** `latin-1` (file gốc không phải UTF-8).
- Sau khi loại NaN và duplicate, dataset còn ~5,169 mẫu, tỉ lệ spam ~13%.
- Chia train/test = 80/20, **stratify theo label** để giữ tỉ lệ spam.

---

## 4. Notebook `Fine_tune_DistilBERT_with_LIME.ipynb`

Notebook gồm **58 cells** (markdown + code), tổ chức theo pipeline tuyến tính từ load dữ liệu → train → evaluate → giải thích. Phần dưới đi từng bước, kèm cell index để dễ tra cứu khi mở notebook.

### Phần 1 — Fine-tune DistilBERT

#### Cell 0 — Markdown mở đầu
Liệt kê 6 bước pipeline tổng quan của Phần 1.

#### Cell 1 — Cài đặt thư viện
```python
!pip install -q transformers datasets evaluate accelerate
!pip install -q lime
```
- `transformers`: model + tokenizer DistilBERT
- `datasets`: lớp `Dataset` để dùng với `Trainer`
- `evaluate`: API tính metric (accuracy, f1)
- `accelerate`: backend tăng tốc training của HuggingFace
- `lime`: dùng cho Phần 2

#### Cell 2 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Notebook giả định chạy trên Colab, dataset và model lưu trên Drive.

#### Cell 3 — Markdown mô tả tiền xử lý
Giải thích cấu trúc CSV gốc: cột `v1` = label, `v2` = text, 3 cột `Unnamed:` rỗng cần loại.

#### Cell 4 — Đọc CSV
```python
CSV_PATH = "/content/drive/MyDrive/DataSpam/spam.csv"
df = pd.read_csv(CSV_PATH, encoding='latin-1')
```
Lưu ý: **bắt buộc** `encoding='latin-1'` — file Kaggle gốc không phải UTF-8, đọc bằng UTF-8 sẽ lỗi `UnicodeDecodeError`. In ra shape, columns, head() để kiểm tra.

#### Cell 5 — Làm sạch dữ liệu
- Giữ 2 cột `v1`, `v2` → đổi tên `label`, `text`
- `dropna()` — loại các dòng rỗng (chủ yếu từ 3 cột `Unnamed:`)
- `drop_duplicates(subset=['text'])` — loại SMS lặp (dataset gốc 5,572 → ~5,169 sau khi loại)
- Map `{'ham': 0, 'spam': 1}` để model dùng được
- In phân bố nhãn và tỉ lệ spam (~12.6%)

#### Cell 6 — EDA: phân bố nhãn và độ dài SMS
Vẽ 2 subplot:
1. Bar chart phân bố ham/spam — cho thấy **dataset rất mất cân bằng** (≈87% ham vs ≈13% spam) → đây là lý do dùng **F1** làm metric chính thay vì accuracy.
2. Histogram phân bố số từ trong SMS, vẽ đường dọc đỏ ở **95th percentile**. Con số này được dùng để chọn `max_length=128` ở bước tokenize — đủ bao phủ 95% SMS, không lãng phí compute padding cho 5% câu dài.

#### Cell 7 — Xem mẫu ham/spam
In 3 ví dụ mỗi loại để có cảm nhận trực quan: ham là tin nhắn cá nhân; spam có pattern rõ (FREE, WIN, claim, số điện thoại, viết hoa).

#### Cell 8 — Train/Test split
```python
train_test_split(..., test_size=0.2, random_state=42, stratify=df['label'])
```
- `test_size=0.2` → 80/20
- `stratify=label` — **bắt buộc** với dataset mất cân bằng, đảm bảo cả 2 tập có cùng tỉ lệ spam
- `random_state=42` — reproducible

#### Cell 9 — Convert sang HuggingFace `Dataset`
`Dataset.from_pandas(...)` — format mà `Trainer` API yêu cầu (tự động hỗ trợ batching, columns selection).

#### Cell 10 — Load tokenizer
```python
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```
Test tokenizer trên 1 câu mẫu để xác minh nó hoạt động: in token IDs và token strings (sẽ thấy `[CLS]`, `[SEP]`, sub-word tokens).

#### Cell 11 — Tokenize toàn bộ dataset
```python
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=128)
train_tokenized = train_dataset.map(tokenize_function, batched=True)
```
- `padding="max_length"` — tất cả câu pad về 128 token (đơn giản, không cần dynamic padding)
- `truncation=True` — câu dài hơn 128 sẽ bị cắt
- `batched=True` — tokenize từng batch để tăng tốc

#### Cell 12 — Markdown giải thích kiến trúc
Note rằng `num_labels=2` nên backbone DistilBERT là pre-trained còn classifier head 2 outputs khởi tạo random.

#### Cell 13 — Load model
```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2,
    id2label={0: "ham", 1: "spam"},
    label2id={"ham": 0, "spam": 1}
)
```
- `id2label`/`label2id`: mapping được lưu cùng model, giúp `pipeline` trả về string `"ham"`/`"spam"` thay vì 0/1
- In tổng số params (~67M) và số trainable params

#### Cell 14 — Định nghĩa metric
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_metric.compute(...)["accuracy"],
        "f1": f1_metric.compute(...)["f1"]
    }
```
Hàm này được `Trainer` gọi tự động sau mỗi epoch eval.

#### Cell 15 — Setup Trainer
```python
training_args = TrainingArguments(
    output_dir="./distilbert-spam",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    save_total_limit=1
)
```
Điểm đáng chú ý:
- `load_best_model_at_end=True` + `metric_for_best_model="f1"` — tự động giữ checkpoint có F1 cao nhất, tránh overfit ở epoch cuối
- `save_total_limit=1` — chỉ giữ 1 checkpoint, tiết kiệm dung lượng Drive
- `report_to="none"` — tắt WandB/TensorBoard logging

#### Cell 16 — Train
```python
trainer.train()
```
Trên Colab GPU T4: ~3–5 phút/epoch, tổng ~15 phút.

#### Cell 17 — Evaluate trên test set
In các metric cuối cùng (accuracy, f1, eval_loss).

#### Cell 18 — Confusion matrix + classification report
```python
predictions = trainer.predict(test_tokenized)
y_pred = np.argmax(predictions.predictions, axis=1)
```
- Vẽ heatmap confusion matrix bằng `seaborn`
- In classification report đầy đủ với `precision`/`recall`/`f1` cho từng lớp

**Kết quả trên test:** Accuracy ≈ **99.03%**, F1 ≈ **0.9612**.

#### Cell 19 — Tách các trường hợp dự đoán SAI
```python
test_results_df['predicted'] = y_pred
test_results_df['actual'] = y_true
wrong = test_results_df[test_results_df['predicted'] != test_results_df['actual']]

fp = wrong[wrong['actual'] == 0]  # ham bị nhầm thành spam
fn = wrong[wrong['actual'] == 1]  # spam bị bỏ sót
```
Đây là **bộ data quan trọng nhất** cho Phần 2 — sẽ được giải thích bằng LIME để tìm hiểu lý do model sai.

#### Cell 20 — Lưu model
```python
SAVE_PATH = "/content/drive/MyDrive/distilbert-spam"
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
```
Lưu cả model và tokenizer để các session sau load lại không cần train.

#### Cell 21 — Demo bằng `pipeline`
Chạy 7 SMS test thủ công (3 ham rõ, 4 spam rõ) qua `pipeline("text-classification")` để xác minh model làm việc end-to-end với input thô.

---

### Phần 2 — Giải thích bằng LIME

#### Cell 22 — Markdown mở đầu Phần 2
Đặt câu hỏi cốt lõi: *"Tại sao model lại đưa ra các dự đoán đó?"* và liệt kê 6 bước giải thích.

#### Cell 23–25 — Setup LIME explainer
```python
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(
    class_names=['ham', 'spam'],
    random_state=42
)
```
- `class_names` quyết định nhãn hiển thị trên HTML output
- `random_state=42` để kết quả reproducible (LIME có yếu tố ngẫu nhiên — xem Bước 5)

#### Cell 26–27 — Định nghĩa `predict_proba` cho LIME
```python
def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True,
                       max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()  # shape (N, 2)
```
**API contract của LIME:**
- Input: `list[str]` các phiên bản nhiễu của câu gốc
- Output: `np.ndarray` shape `(N, num_classes)` — xác suất từng class
- `with torch.no_grad()` — tắt gradient (không train, chỉ inference)
- `model.eval()` — tắt dropout/batchnorm randomness

#### Cell 28–31 — Bước 2: Giải thích 1 SMS spam điển hình
Chọn câu `"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005"` và:
1. Dự đoán bằng model gốc (in `P(ham)`, `P(spam)`)
2. Gọi `explainer.explain_instance(text, predict_proba, num_features=10, num_samples=1000)`:
   - `num_samples=1000` — tạo 1000 phiên bản nhiễu
   - `num_features=10` — top 10 từ quan trọng nhất
3. `explanation.show_in_notebook(text=True)` — render HTML inline
4. In dạng list ASCII bar chart cho dễ đọc trên console

#### Cell 32–33 — Bước 3: Chọn các SMS dự đoán ĐÚNG
Tách `correct_ham` và `correct_spam` từ `test_results_df`, lọc các câu có 8–25 từ (đủ ngữ cảnh để LIME phân tích, không quá ngắn cũng không quá dài), lấy 3 mẫu đại diện mỗi loại.

#### Cell 34–35 — Hàm helper `explain_and_print`
Hàm tiện ích nhận `text`, `true_label`, `predicted_label`, gọi LIME (`num_features=8`, `num_samples=1000`) và in:
- Header với SMS, nhãn thật, dự đoán, xác suất
- Bảng top 8 từ + trọng số + hướng (→ SPAM hoặc → HAM) + bar chart ASCII

Trả về object `exp` để dùng tiếp ở các bước sau.

#### Cell 36–37 — Giải thích 3 SMS HAM đúng
Lưu kết quả vào `ham_explanations` (list các tuple `(text, exp)`).

#### Cell 38–39 — Giải thích 3 SMS SPAM đúng
Lưu vào `spam_explanations`.

#### Cell 40 — Markdown câu hỏi suy nghĩ
3 câu hỏi để viết vào báo cáo: model có học pattern hợp lý không, có pattern đáng nghi ngờ không, có khớp trực giác con người không.

#### Cell 41–42 — Bước 4: Phân tích các trường hợp SAI ⭐
**Đây là phần ăn điểm cao nhất**. Lấy lại `fp_cases` và `fn_cases` từ `wrong` DataFrame (Cell 19).

#### Cell 43–44 — False Positives (ham → spam)
Chạy LIME trên TẤT CẢ FP cases để trả lời: *"Tại sao model nói tin nhắn vô hại này là spam?"* Lưu vào `fp_explanations`.

#### Cell 45–46 — False Negatives (spam → ham)
Chạy LIME trên TẤT CẢ FN cases để trả lời: *"Spammer dùng kỹ thuật gì để né model?"* Lưu vào `fn_explanations`.

#### Cell 47 — Markdown insights
Gợi ý phân tích cho báo cáo: liên hệ với thí nghiệm **Husky vs Wolf** trong paper LIME gốc — model có học "shortcut" không hợp lý không.

#### Cell 48–50 — Bước 5: Test tính ổn định của LIME
```python
explainer_unstable = LimeTextExplainer(class_names=['ham', 'spam'])
# KHÔNG có random_state → mỗi lần khác nhau
for i in range(3):
    exp = explainer_unstable.explain_instance(...)
    ...
```
- Chạy 3 lần trên cùng câu `"Free entry to win iPhone now! Click here"`
- So sánh trọng số từng từ qua 3 lần chạy
- Tính **standard deviation** của trọng số → đo mức bất ổn định
- Quan sát điển hình: top 3-5 từ ổn định, từ ít quan trọng dao động nhiều

#### Cell 51 — Markdown thảo luận hạn chế
Đề xuất cách giảm bất ổn định:
- Tăng `num_samples` (1000 → 5000) — chính xác hơn nhưng chậm hơn
- Cố định `random_state` — reproducible
- Hoặc dùng SHAP — có nền tảng toán học (Shapley values) chặt hơn

#### Cell 52–53 — Bước 6: Lưu HTML output
```python
OUTPUT_DIR = "/content/drive/MyDrive/lime_explanations"
for i, (text, exp) in enumerate(spam_explanations):
    exp.save_to_file(f"{OUTPUT_DIR}/correct_spam_{i+1}.html")
# tương tự cho ham, FP, FN
```
Tạo ra 16 file HTML (3 ham + 3 spam + 5 FP + 5 FN) — chính là các file trong thư mục `lime_explanations/` của repo.

#### Cell 54–56 — Bước 7: Pattern analysis (giống SP-LIME)
```python
spam_words = Counter()
for text, exp in spam_explanations:
    for word, weight in exp.as_list():
        if weight > 0:                      # chỉ lấy từ đẩy về spam
            spam_words[word] += weight      # cộng dồn trọng số

ham_words = Counter()
for text, exp in ham_explanations:
    for word, weight in exp.as_list():
        if weight < 0:
            ham_words[word] += abs(weight)
```
Ý tưởng: gộp trọng số của cùng một từ qua nhiều SMS → từ nào **liên tục** đẩy về spam/ham là **pattern toàn cục** model đã học.

Cell 56 vẽ biểu đồ ngang top 10 từ mỗi hướng và lưu thành `top_words_analysis.png`. Đây là cách xấp xỉ **Submodular Pick LIME (SP-LIME)** trong paper gốc — chuyển từ giải thích **cục bộ** sang **toàn cục**.

#### Cell 57 — Markdown tổng kết
Checklist các bước đã hoàn thành + cấu trúc báo cáo gợi ý 5 phần (Giới thiệu, Phương pháp, Thực nghiệm, Thảo luận, Kết luận) + mẹo ghi điểm cao.

---

### Tóm tắt biến quan trọng giữ giữa các cell

| Biến | Tạo ở cell | Dùng ở cell | Nội dung |
|---|---|---|---|
| `df` | 4–5 | 6–8 | DataFrame đã clean, có `text` và `label` |
| `train_df`, `test_df` | 8 | 9, 19 | Split 80/20 stratified |
| `train_tokenized`, `test_tokenized` | 11 | 15–18 | Dataset đã tokenize |
| `model`, `tokenizer` | 10, 13 | 15, 21, 27+ | Model fine-tune và tokenizer |
| `trainer` | 15 | 16–18, 20 | HuggingFace Trainer |
| `y_pred`, `y_true` | 18 | 19 | Mảng predictions để tính metric / lọc lỗi |
| `test_results_df` | 19 | 33, 42 | DataFrame test + cột `predicted`/`actual` |
| `wrong`, `fp`, `fn` | 19, 42 | 44, 46 | Các trường hợp model sai |
| `explainer` | 25 | 29, 35 | `LimeTextExplainer` cố định seed |
| `predict_proba` | 27 | 29, 35, 49 | Hàm bridge giữa model và LIME |
| `ham_explanations`, `spam_explanations` | 37, 39 | 53, 55 | List `(text, exp)` cho dự đoán đúng |
| `fp_explanations`, `fn_explanations` | 44, 46 | 53 | List `(text, exp)` cho dự đoán sai |

---

## 5. LIME hoạt động như thế nào (tóm tắt)

LIME giải thích **từng dự đoán riêng lẻ** (local explanation) bằng 4 bước:

1. **Perturb:** tạo ~1000 phiên bản nhiễu của câu gốc bằng cách xóa ngẫu nhiên một số từ.
2. **Predict:** đưa toàn bộ các phiên bản này qua model gốc (DistilBERT) lấy `P(spam)`.
3. **Weight:** mỗi phiên bản được gán trọng số theo độ giống câu gốc (cosine distance).
4. **Fit linear model:** train một mô hình tuyến tính (Ridge) đơn giản → hệ số của mỗi từ chính là **mức độ ảnh hưởng** đẩy dự đoán về `spam` (dương) hoặc `ham` (âm).

> Vì bước perturb ngẫu nhiên, kết quả 2 lần chạy có thể khác nhau → notebook có riêng Bước 5 để đo độ ổn định và thảo luận hạn chế này (so sánh với SHAP).

---

## 6. Output trong `lime_explanations/`

17 file, sinh ra từ Bước 6 của notebook:

| File | Ý nghĩa |
|---|---|
| `correct_ham_{1..3}.html` | 3 SMS ham được dự đoán **đúng** + highlight từ đẩy về ham |
| `correct_spam_{1..3}.html` | 3 SMS spam được dự đoán **đúng** + highlight từ đẩy về spam |
| `false_positive_{1..5}.html` | SMS ham bị model nhận **nhầm thành spam** — trả lời: "tại sao model nghi ngờ tin nhắn vô hại này?" |
| `false_negative_{1..5}.html` | SMS spam bị model **bỏ sót** — trả lời: "spammer đã dùng kỹ thuật gì để né model?" |
| `top_words_analysis.png` | 2 biểu đồ ngang: top 10 từ đẩy về SPAM (đỏ) và top 10 từ đẩy về HAM (xanh) |

Mỗi file HTML có thể mở trực tiếp trên trình duyệt — sẽ hiện:
- Thanh xác suất `[ham | spam]` của model
- Bảng các từ và trọng số (đỏ = đẩy về spam, xanh = đẩy về ham)
- Văn bản gốc với từ được tô màu theo hướng tác động

---

## 7. Cách chạy lại

Notebook được thiết kế để chạy trên **Google Colab** (có dòng `from google.colab import drive`).

### Trên Colab
1. Upload `dataset/spam.csv` lên Drive: `MyDrive/DataSpam/spam.csv`
2. Mở `Fine_tune_DistilBERT_with_LIME.ipynb` trong Colab
3. Bật GPU runtime (Runtime → Change runtime type → GPU)
4. Run all — model lưu vào `MyDrive/distilbert-spam`, HTML lưu vào `MyDrive/lime_explanations`

### Local (nếu muốn)
- Sửa `CSV_PATH` trong cell 4 thành đường dẫn local: `dataset/spam.csv`
- Bỏ 2 cell `drive.mount(...)` và đổi `SAVE_PATH` / `OUTPUT_DIR` thành đường dẫn local
- Cài thư viện:
  ```bash
  pip install transformers datasets evaluate accelerate lime scikit-learn matplotlib seaborn pandas
  ```
- Cần GPU để huấn luyện ở tốc độ chấp nhận được (CPU vẫn chạy được nhưng rất chậm).

---

## 8. Hyperparameters chính

| Tham số | Giá trị | Lý do |
|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased` | Nhẹ hơn BERT 40%, nhanh hơn 60%, giữ ~97% chất lượng |
| `max_length` | 128 | Bao phủ ~95% SMS (theo phân tích độ dài) |
| `num_train_epochs` | 3 | Đủ hội tụ với dataset nhỏ, tránh overfit |
| `batch_size` | 16 train / 32 eval | Vừa với GPU Colab T4 |
| `learning_rate` | `2e-5` | Chuẩn cho fine-tune BERT-family |
| `weight_decay` | 0.01 | Regularization |
| `num_samples` (LIME) | 1000 | Cân bằng giữa độ chính xác và tốc độ |
| `num_features` (LIME) | 8–10 | Đủ để xem pattern, không nhiễu |

---

## 9. Hướng phát triển

- **Stability:** so sánh LIME với **SHAP** (Shapley values) — ổn định hơn về mặt toán học.
- **Global explanation:** triển khai SP-LIME đầy đủ để chọn tập SMS đại diện thay vì chỉ aggregate top từ.
- **Robustness:** test model trên adversarial SMS (thay ký tự `free` → `fr3e`, `win` → `w!n`).
- **Calibration:** kiểm tra xác suất model có well-calibrated không (reliability diagram).
- **Multilingual:** mở rộng sang SMS tiếng Việt (`distilbert-base-multilingual-cased` hoặc `xlm-roberta`).

---

## 10. Tham khảo

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016.
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS Workshop.
- SMS Spam Collection Dataset — Kaggle / UCI Machine Learning Repository.
