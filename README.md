# PTB-XL 多模態 ECG 診斷分類訓練框架

影像模態（ECG 印刷影像）+ 年齡模態（輔助資訊）→ 診斷 superclass 分類（NORM / MI / STTC / CD / HYP）

延續原本影像分類的模組化架構（block-based），每個模組獨立、可透過 `CONFIG` 開關切換，方便在 Kaggle Notebook 中逐格執行與除錯。

---

## 1. 任務說明

- **輸入**：ECG 印刷影像（PTB-XL 影像化版本）+ 病患年齡（數值）
- **輸出**：5 個診斷大分類之一
- **標籤定義**（PTB-XL `scp_statements.csv` 的 diagnostic superclass）：

| 標籤 | 全名 | 意義 |
|---|---|---|
| NORM | Normal ECG | 正常心電圖 |
| MI | Myocardial Infarction | 心肌梗塞 |
| STTC | ST/T Change | ST-T 段異常 |
| CD | Conduction Disturbance | 傳導障礙 |
| HYP | Hypertrophy | 心臟肥大 |

目前採**單標籤模式**（`SINGLE_LABEL_ONLY=True`）：只保留 `diagnostic_superclass` 剛好對應到單一類別的紀錄，簡化成單標籤分類問題。

---

## 2. 資料集

| 用途 | Kaggle Dataset | 內容 |
|---|---|---|
| 影像 | `bjoernjostein/ptb-xl-ecg-image-gmc2024` | ECG 印刷影像，命名格式 `{ecg_id:05d}_{lr\|hr}-{idx}.png`，依 `ecg_id` 千位數分組存放於子資料夾（如 `12345` → `12000/` 資料夾） |
| Metadata | `khyeh0719/ptb-xl-dataset` | 內含 `ptbxl_database.csv`（age、sex、scp_codes 等）與 `scp_statements.csv`（診斷代碼對應表） |

**使用前務必確認實際掛載路徑**，Kaggle 掛載路徑常見格式為：
```
/kaggle/input/datasets/{owner}/{dataset-slug}/...
```
不同帳號/版本可能不同，建議先用以下程式碼確認：
```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f in ("ptbxl_database.csv", "scp_statements.csv"):
            print("找到:", os.path.join(root, f))
```

---

## 3. 環境需求

- Kaggle Notebook（建議開啟 GPU accelerator）
- 主要套件：`torch`、`timm`、`torchvision`、`pandas`、`scikit-learn`、`matplotlib`、`Pillow`

---

## 4. 程式架構（Block A–S）

| Block | 內容 | 對應開關 |
|---|---|---|
| A | 全域 CONFIG（路徑、超參數、開關） | 所有設定集中於此 |
| B | 隨機種子固定 | — |
| C | PTB-XL metadata 讀取 + diagnostic superclass 聚合 | `SINGLE_LABEL_ONLY` |
| D | 年齡清理（含缺值濾除）+ 影像路徑展開 | `AGE_CLIP_MAX`、`MAX_IMG_PER_RECORD` |
| E | Patient-level train/val/test 切分（避免資料洩漏） | — |
| F | 年齡標準化（僅用 train set 統計量） | — |
| G | 影像增強（transform） | — |
| H | 多模態 Dataset（回傳 image + age + label） | `USE_AGE_MODALITY` |
| I | WeightedRandomSampler（處理類別不平衡） | `USE_WEIGHTED_SAMPLER` |
| J | DataLoader 建構 | — |
| K | 多模態模型（image encoder + age encoder + late fusion） | `USE_AGE_MODALITY`、`AGE_EMBED_DIM` |
| L | Loss（class-weighted CrossEntropy / Focal Loss） | `USE_CLASS_WEIGHTED_LOSS`、`USE_FOCAL_LOSS` |
| M | Optimizer（backbone / head 分層 learning rate） | `LR_BACKBONE`、`LR_HEAD` |
| N | LR Scheduler（CosineAnnealingLR） | — |
| O | Mixup（僅作用於影像分支） | `USE_MIXUP` |
| P | 訓練 / 驗證迴圈（含 batch-level 進度顯示、precision/recall/f1/accuracy 計算） | `EARLY_STOP_METRIC` |
| Q | Grad-CAM（僅影像分支，年齡分支無法用此方式視覺化） | `USE_GRADCAM` |
| R | 主流程（`main()`，串接所有模組） | — |
| S | 視覺化（自動產出訓練曲線、混淆矩陣、各類別指標等圖表） | — |

---

## 5. 執行方式

1. 依序執行 Block A → S（讓所有函式定義載入記憶體）
2. 執行：
   ```python
   model, test_metrics = main(CONFIG)
   ```
3. 訓練完成後，`test_metrics["report"]` 印出完整 classification report，並自動產出以下圖表至 `CONFIG["OUTPUT_DIR"]`：

| 檔名 | 內容 |
|---|---|
| `00_results_grid.png` | 訓練過程六宮格總覽（train_loss / val_loss / precision / recall / f1 / accuracy 隨 epoch 走勢，仿 YOLO `results.png` 風格） |
| `01_training_curves.png` | Loss 與 F1 曲線放大版 |
| `02_confusion_matrix.png` | 混淆矩陣（數量版 + 百分比正規化版） |
| `03_per_class_metrics.png` | 各類別 Precision / Recall / F1 長條圖 |
| `04_imbalance_vs_recall.png` | 類別樣本數 vs Recall 對照圖 |
| `best_model.pth` | 驗證集 F1 最佳的模型權重 |
| `sample_gradcam.npy` | 一筆 test 樣本的 Grad-CAM 熱力圖陣列 |

**建議用「Save Version → Save & Run All (Commit)」執行**，而非互動模式掛著等，避免 session 過期或斷線導致 `/kaggle/working` 底下的訓練結果被清空。Commit 執行完的 Output 會被永久保存，不受 session 過期影響。

---

## 6. 重要注意事項

- **`age` 欄位有缺值**：PTB-XL 有少量紀錄缺少年齡（約 38 筆），Block D 的 `clean_age()` 會自動濾除，避免污染 `AgeScaler` 造成 mean/std 變成 `nan`
- **`ecg_id` 與影像檔名的對應**：影像檔名為 5 位數補零 + rate 後綴 + image index（如 `00001_lr-0.png`），資料夾依千位數分組，Block D 的 `get_ptbxl_folder()` 會自動處理路徑對應
- **Patient-level 切分**：務必用 `patient_id` 而非 `ecg_id` 做 train/val/test 切分，避免同一病人的多張合成影像同時出現在不同 split，造成資料洩漏
- **`NUM_WORKERS=0` 建議值**：Kaggle 環境下較穩定，可避免中斷 cell 後殘留 DataLoader worker 導致的 `AssertionError` 清理雜訊

---

## 7. Ablation 實驗結果（年齡模態是否有幫助）

| 指標 | 無年齡模態（純影像） | 有年齡模態（多模態） | 差異 |
|---|---|---|---|
| Macro F1 | 0.58 | 0.59 | +0.01 |
| Macro Precision | 0.56 | 0.57 | +0.01 |
| Macro Recall | 0.67 | 0.66 | −0.01 |
| Accuracy | 0.62 | 0.63 | +0.01 |

**結論**：目前的年齡融合方式（scalar age → 小型 MLP → late fusion concat）對這個 5 類別分類任務的貢獻幾乎可忽略，差異落在訓練隨機性造成的正常波動範圍內。可能原因與後續改進方向：

- 年齡以 scalar 形式輸入資訊量太弱，可嘗試離散化分組 embedding 或更強的融合方式（如 FiLM）
- NORM/MI/STTC/CD/HYP 這五類主要靠 ECG 波形型態區分，年齡本身可能是弱相關的人口學變項，不一定是這個任務的有效輔助模態

---

## 8. 已知的模型表現問題

- **NORM 類別 recall 偏低（約 0.55–0.57）**：即使樣本數最多，仍有 20%+ 被誤判成 MI/STTC，推測與 `WeightedRandomSampler` 拉高稀有類別抽樣頻率、間接壓低 NORM 有效訓練頻率有關
- **HYP 類別 precision 偏低（約 0.33–0.37）**：其他類別容易被誤判成 HYP，是不平衡處理策略的副作用（trade-off：稀有類別 recall 提升的代價）
- **best epoch 之後略有過擬合**：val_loss 觸底後緩慢回升、val_f1 出現鋸齒震盪，可考慮降低 `LR_BACKBONE` 或縮小 `EARLY_STOP_PATIENCE` 改善訓練穩定性

---

## 9. 待辦與未來方向

- [ ] Checkpoint / Resume 機制（訓練中斷後可從上次 epoch 接續，尚未整合進本檔案，僅完成設計討論）
- [ ] 嘗試年齡分組 embedding 取代 scalar 輸入
- [ ] 遷移至 KD (Kawasaki Disease) ECG 資料集（ZZU pECG dataset），驗證年齡模態在 KD 診斷任務上是否有更強的鑑別力（KD 好發於特定年齡層，訊息量可能高於一般成人心臟病診斷任務）
