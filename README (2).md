# PTB-XL 多模態 ECG 診斷分類訓練框架

影像模態（ECG 印刷影像）+ 人口學模態（年齡 + 性別）→ 診斷 superclass 分類（NORM / MI / STTC / CD / HYP）

模組化架構（Block A–S），每個模組獨立、可透過 `CONFIG` 開關切換，方便在 Kaggle Notebook 逐格執行、除錯與 ablation 實驗。

---

## 1. 任務與標籤

| 標籤 | 全名 | 意義 |
|---|---|---|
| NORM | Normal ECG | 正常心電圖 |
| MI | Myocardial Infarction | 心肌梗塞 |
| STTC | ST/T Change | ST-T 段異常 |
| CD | Conduction Disturbance | 傳導障礙 |
| HYP | Hypertrophy | 心臟肥大 |

採單標籤模式（`SINGLE_LABEL_ONLY=True`）：只保留 `diagnostic_superclass` 剛好對應單一類別的紀錄。

---

## 2. 資料集

| 用途 | Kaggle Dataset | 內容 |
|---|---|---|
| 影像 | `bjoernjostein/ptb-xl-ecg-image-gmc2024` | 命名格式 `{ecg_id:05d}_{lr\|hr}-{idx}.png`，依千位數分組存放 |
| Metadata | `khyeh0719/ptb-xl-dataset` | `ptbxl_database.csv`（age/sex/scp_codes）、`scp_statements.csv`（診斷代碼對照表） |

**注意**：PTB-XL 影像頂端印有病人資訊文字（日期、檔案路徑含編號、性別、年齡），Block G 會自動遮蔽避免資訊洩漏，詳見下方 Block G 說明。

---

## 3. Block 總覽表

| Block | 內容 | 對應開關 |
|---|---|---|
| A | 全域 CONFIG（路徑、超參數、開關） | 所有設定集中管理 |
| B | 隨機種子固定 | — |
| C | PTB-XL metadata 讀取 + diagnostic superclass 聚合 | `SINGLE_LABEL_ONLY` |
| D | 年齡/性別清理 + 影像路徑展開 | `AGE_CLIP_MAX` |
| E | Patient-level train/val/test 切分（防資料洩漏） | — |
| F | 年齡 Z-score 標準化 | — |
| G | 影像增強 + 病人資訊遮罩 | `MASK_TOP_RATIO` |
| H | 多模態 Dataset（image + demo + label） | `USE_AGE_MODALITY`、`USE_SEX_MODALITY` |
| I | WeightedRandomSampler（類別不平衡處理） | `USE_WEIGHTED_SAMPLER` |
| J | DataLoader 建構 | — |
| K | 多模態模型（image encoder + demo encoder + late fusion） | `DEMO_EMBED_DIM` |
| L | Loss（class-weighted CE / Focal Loss） | `USE_FOCAL_LOSS` |
| M | Optimizer（backbone/head 分層學習率） | `LR_BACKBONE`、`LR_HEAD` |
| N | LR Scheduler（ReduceLROnPlateau） | — |
| O | Mixup（僅影像分支，預設關閉） | `USE_MIXUP` |
| P | 訓練/驗證迴圈（含 precision/recall/f1/accuracy） | `EARLY_STOP_METRIC` |
| Q | Grad-CAM（含視覺化圖片產出） | `USE_GRADCAM` |
| R | 主流程（`main()`，串接所有模組） | — |
| S | 視覺化（訓練曲線、混淆矩陣、彙總表等） | — |

---

## 4. 逐 Block 技術詳解

### Block A — 全域設定（CONFIG）

所有超參數與開關集中管理，是整個 pipeline 唯一需要修改的地方（正常使用情境下）。

- **關注點分離**：路徑、任務設定、模型架構、訓練超參數、不平衡處理、資料增強、Grad-CAM、early stopping 分區塊管理
- **Ablation 開關設計**：`USE_AGE_MODALITY`、`USE_SEX_MODALITY` 讓同一份程式碼可以做「有/無某個模態」的對照實驗

### Block B — 隨機種子固定

固定 Python、NumPy、PyTorch（CPU+GPU）的隨機數來源，並關閉 cuDNN 自動優化搜尋，換取**實驗可重現性**——沒有這一步，同樣的 CONFIG 兩次執行結果可能不同，讓 ablation 比較失去意義。

### Block C — Metadata 讀取與診斷聚合

- `ast.literal_eval` 安全解析 `scp_codes` 字串成 dict（比 `eval()` 安全）
- **診斷代碼聚合**：把細部診斷代碼透過 `scp_statements.csv` 映射到5個superclass大類
- **多標籤過濾為單標籤**：只保留剛好對應單一 superclass 的紀錄

### Block D — 年齡/性別清理 + 影像路徑展開

- **缺值處理**：`clean_age()`、`clean_sex()` 都直接濾除缺值紀錄（PTB-XL約38筆缺年齡）
- **邊界值處理**：PTB-XL 對 >89歲統一設為300歲（HIPAA去識別化），`clip()` 裁切回合理範圍
- **確定性路徑映射**：`get_ptbxl_folder()` 用整數除法計算資料夾分組
- **長表格式轉換**：把「一筆病人紀錄」展開成「一筆影像」的長表

### Block E — Patient-level 資料切分

- **`GroupShuffleSplit`**：以 `patient_id` 為切分單位，保證同一病人的所有紀錄都在同一個split，避免資料洩漏
- **兩階段切分**：先切test set，再從剩餘資料切val set，並用 `assert` 硬性檢查三個split互斥

### Block F — 年齡 Z-score 標準化

```
z = (age - mean) / std
```

- `AgeScaler.fit()` 只用 train set 計算 mean/std，val/test 只呼叫 `transform()`，避免資料洩漏

### Block G — 影像資料增強 + 病人資訊遮罩

- **`MaskTopRegion`**：遮蔽影像最上方20%區域（可調），避免模型學到影像上列印的病人編號、路徑、年齡、性別等文字資訊——這些資訊如果沒遮蔽，理論上模型可能透過影像文字「偷看」到原本應該獨立輸入的人口學資料，讓多模態架構的模態獨立性失真。train/eval 都套用，確保輸入分布一致
- **保守的增強策略**：`RandomRotation(5)`、`ColorJitter` 微調亮度對比度，刻意不用翻轉類增強（會破壞ECG波形的臨床意義）
- **`RandomErasing`**：訓練時隨機遮擋小區塊，抑制過擬合
- **ImageNet標準化**：因為 backbone 用ImageNet預訓練權重，輸入分布需要跟預訓練時一致

### Block H — 多模態 Dataset

- `__getitem__` 一次回傳三個張量（image, demo, label）
- **動態特徵組合**：`demo_values` 依開關動態組裝成1維或2維向量
- 性別（二元類別）不做z-score，直接當數值輸入（0/1）

### Block I — WeightedRandomSampler

- **逆頻率加權抽樣**：`weight = total_samples / class_count`，樣本數越少的類別被抽中機率越高
- `replacement=True` 允許同一筆資料在一個epoch內被重複抽樣

### Block J — DataLoader 建構

只有 train_loader 套用 WeightedRandomSampler，val/test 維持原始分布，確保評估反映真實世界表現。

### Block K — 多模態融合模型

- **雙分支編碼器**：影像走 CNN backbone（`timm.create_model`），人口學特徵走小型MLP
- **晚期融合（Late Fusion）**：`torch.cat([img_feat, demo_feat], dim=1)`，兩分支輸出在分類頭之前才拼接
- **遷移學習**：載入ImageNet預訓練權重，`num_classes=0`拿掉原分類層
- **動態維度適配**：`img_feat_dim = self.img_encoder.num_features` 自動抓取，換backbone不用改其他程式碼
- **凍結/解凍機制**：訓練初期凍結backbone，只訓練新模組，穩定後才解凍一起微調

### Block L — Loss Function

- **類別加權交叉熵**：讓稀有類別的錯誤在loss計算時被放大
- **權重截斷**：`clamp(min=0.5, max=5.0)` 避免極端不平衡算出過大權重
- **Focal Loss（選用）**：讓模型聚焦在難分類樣本上

### Block M — Optimizer

- **分層學習率**：預訓練backbone用較小LR（微調），新模組用較大LR（從頭學）
- **AdamW**：解耦權重衰減，比傳統Adam+L2更穩定

### Block N — LR Scheduler

- **`ReduceLROnPlateau`**：監控 `val_f1_macro`，連續 `patience=3` 個epoch沒進步就LR減半，直到 `min_lr`。相對於固定排程的方式，這是依模型實際表現動態調整
- **注意**：scheduler的patience要明顯小於 `EARLY_STOP_PATIENCE`，否則兩者同時觸發，LR才剛降訓練就被停止，scheduler等於沒發揮作用

### Block O — Mixup（預設關閉）

- 特徵空間線性插值增強，本專案因ECG印刷影像的像素混合可能產生不具臨床意義的波形疊加，預設關閉
- 人口學特徵不參與混合，維持原值傳遞

### Block P — 訓練與驗證迴圈

- `torch.no_grad()` 停用梯度計算，節省驗證階段資源
- **Macro average 指標**：precision/recall/f1都採macro平均，更真實反映稀有類別表現，不被多數類別掩蓋
- Batch-level進度顯示，方便確認長時間訓練是否正常執行

### Block Q — Grad-CAM（可解釋性）

- **Grad-CAM**：透過forward/backward hook抓取目標層的activation與梯度，計算出反映「模型關注哪些影像區域」的熱力圖
- **視覺化輸出**：`plot_gradcam_single()` 產出原圖/純熱力圖/疊圖三合一，`generate_gradcam_examples()` 每個類別各挑樣本自動產圖
- **侷限**：僅適用於CNN空間特徵，只能解釋影像分支貢獻，人口學分支無法用此方式視覺化

### Block R — 主流程（`main()`）

- **端到端pipeline編排**：資料處理→模型建構→訓練→best checkpoint回載→test評估→自動產圖
- **Best checkpoint選擇**：追蹤驗證集表現最好的epoch權重，而非用最後一個epoch（避免用到已過擬合的權重）
- **Early stopping**：連續 `EARLY_STOP_PATIENCE` 個epoch沒進步就提前終止

### Block S — 視覺化

- **CJK字型動態載入**，避免中文標籤顯示異常
- **YOLO風格六宮格總覽**（`00_results_grid.png`）：train/val loss、precision/recall/f1/accuracy隨epoch走勢
- **混淆矩陣雙版本**（`02_confusion_matrix.png`）：數量版+列正規化百分比版
- **各類別指標長條圖**（`03_per_class_metrics.png`）與**不平衡對照圖**（`04_imbalance_vs_recall.png`）
- **整體彙總表**（`05_summary_table.png`/`.csv`）：各類別指標 + Accuracy/Macro Avg/Weighted Avg 合一呈現
- **`print_overall_summary()`**：訓練結束印出簡潔文字版整體結果（Accuracy/Precision/Recall/F1），方便快速複製紀錄

---

## 5. 執行方式

1. 依序執行 Block A → S
2. 執行 `model, test_metrics = main(CONFIG)`
3. 建議用 **Save Version → Save & Run All (Commit)** 執行，避免 session 過期導致輸出被清空

輸出檔案：`00_results_grid.png`、`01_training_curves.png`、`02_confusion_matrix.png`、`03_per_class_metrics.png`、`04_imbalance_vs_recall.png`、`05_summary_table.png/csv`、`gradcam_{class}_0.png` × 5、`best_model.pth`

---

## 6. 已知結果與問題

- **年齡模態 ablation**：初期版本對5類別分類任務貢獻約±0.01 F1，效果不顯著，晚期融合強度可能是限制因素
- **LR/Scheduler調整後**：訓練穩定性明顯改善，val_loss不再發散，但整體accuracy略降，稀有類別（HYP、MI）recall大幅提升——是合理的trade-off，而非退步
- **NORM/HYP的precision-recall trade-off**：WeightedRandomSampler讓稀有類別recall提升的同時，也讓多數類別容易被誤判

---

## 7. 待辦事項

- [ ] Checkpoint / Resume 機制（尚未整合進本檔案）
- [ ] 年齡分組 embedding 或 FiLM 融合，取代目前的 scalar + concat 方式
- [ ] 遷移至 KD (Kawasaki Disease) ECG 資料集（ZZU pECG dataset）
