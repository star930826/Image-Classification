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

**注意**：PTB-XL 影像頂端印有病人資訊文字（日期、檔案路徑含編號、性別、年齡），Block G 會自動遮蔽避免資訊洩漏。

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
| G | 影像增強（RandomAffine整合式）+ 病人資訊遮罩 + 增強效果預覽 | `MASK_TOP_RATIO`、`AUG_*` |
| H | 多模態 Dataset（image + demo + label） | `USE_AGE_MODALITY`、`USE_SEX_MODALITY` |
| I | WeightedRandomSampler（類別不平衡處理） | `USE_WEIGHTED_SAMPLER` |
| J | DataLoader 建構 | — |
| K | 多模態模型（image encoder + demo encoder + late fusion） | `DEMO_EMBED_DIM`、`BACKBONE` |
| L | Loss（class-weighted CE / Focal Loss） | `USE_FOCAL_LOSS` |
| M | Optimizer（backbone/head 分層學習率） | `LR_BACKBONE`、`LR_HEAD` |
| N | LR Scheduler（可切換 ReduceLROnPlateau / CosineAnnealingLR） | `SCHEDULER_TYPE` |
| O | Mixup（僅影像分支，預設關閉） | `USE_MIXUP` |
| P | 訓練/驗證迴圈（含train/val accuracy、precision/recall/f1） | `EARLY_STOP_METRIC`、`VERBOSE_BATCH` |
| Q | Grad-CAM（含視覺化圖片產出） | `USE_GRADCAM`、`GRADCAM_TARGET_LAYER` |
| R | 主流程（`main()`，含結構化訓練log顯示） | — |
| S | 視覺化（訓練曲線、混淆矩陣、彙總表等） | — |

---

## 4. 逐 Block 技術詳解

### Block A — 全域設定（CONFIG）

所有超參數與開關集中管理，是整個 pipeline 唯一需要修改的地方（正常使用情境下）。

- **關注點分離**：路徑、任務設定、模型架構、資料增強、訓練超參數、不平衡處理、Grad-CAM、early stopping 分區塊管理
- **Ablation 開關設計**：`USE_AGE_MODALITY`、`USE_SEX_MODALITY` 讓同一份程式碼可以做「有/無某個模態」的對照實驗
- **⚠️ 換 `BACKBONE` 時務必同步檢查 `GRADCAM_TARGET_LAYER`**：不同模型家族的內部層命名不同（ResNet系列用`"layer4"`，ConvNeXt系列用`"stages.3"`），兩者沒對應好，訓練/存檔/畫圖都會正常完成，但最後一步Grad-CAM會直接crash

### Block B — 隨機種子固定

固定 Python、NumPy、PyTorch（CPU+GPU）的隨機數來源，並關閉 cuDNN 自動優化搜尋，換取**實驗可重現性**。

### Block C — Metadata 讀取與診斷聚合

- `ast.literal_eval` 安全解析 `scp_codes` 字串成 dict
- **診斷代碼聚合**：把細部診斷代碼透過 `scp_statements.csv` 映射到5個superclass大類
- **多標籤過濾為單標籤**：只保留剛好對應單一 superclass 的紀錄

### Block D — 年齡/性別清理 + 影像路徑展開

- **缺值處理**：`clean_age()`、`clean_sex()` 都直接濾除缺值紀錄
- **邊界值處理**：PTB-XL 對 >89歲統一設為300歲，`clip()` 裁切回合理範圍
- **確定性路徑映射**：`get_ptbxl_folder()` 用整數除法計算資料夾分組

### Block E — Patient-level 資料切分

- **`GroupShuffleSplit`**：以 `patient_id` 為切分單位，避免同一病人的紀錄跨split造成資料洩漏
- 用 `assert` 硬性檢查三個split互斥

### Block F — 年齡 Z-score 標準化

```
z = (age - mean) / std
```
`AgeScaler.fit()` 只用 train set 計算 mean/std，避免資料洩漏。

### Block G — 影像資料增強 + 病人資訊遮罩 + 增強效果預覽

- **`MaskTopRegion`**：遮蔽影像最上方一定比例區域，避免模型學到病人編號、路徑、年齡、性別等文字資訊（這些資訊若不遮蔽，可能讓模型透過影像文字「偷看」到原本應獨立輸入的人口學資料，破壞多模態架構的模態獨立性）
- **`RandomAffine`整合式仿射變換**：旋轉、平移、縮放、剪切合併在同一次操作內完成，避免多次獨立變換造成重複插值的畫質耗損；邊界填充白色符合心電圖背景
- **`RandomErasing`**：訓練時隨機遮擋小區塊，抑制過擬合
- **`visualize_augmentations()`**：訓練前可先產出「原圖/遮罩後/N張隨機增強範例」並列圖，確認增強強度合理，不用等訓練跑完才發現設定不對
- 所有增強參數集中在 CONFIG 的 `AUG_*` 系列，調整強度不用改程式邏輯

### Block H — 多模態 Dataset

- `__getitem__` 一次回傳三個張量（image, demo, label）
- **動態特徵組合**：`demo_values` 依開關動態組裝成1維或2維向量
- 性別（二元類別）不做z-score，直接當數值輸入（0/1）

### Block I — WeightedRandomSampler

- **逆頻率加權抽樣**：樣本數越少的類別被抽中機率越高
- `replacement=True` 允許同一筆資料在一個epoch內被重複抽樣

### Block J — DataLoader 建構

只有 train_loader 套用 WeightedRandomSampler，val/test 維持原始分布，確保評估反映真實世界表現。

### Block K — 多模態融合模型

- **雙分支編碼器**：影像走 CNN backbone（`timm.create_model`，通用寫法，換模型只需改CONFIG的`BACKBONE`字串），人口學特徵走小型MLP
- **晚期融合（Late Fusion）**：`torch.cat([img_feat, demo_feat], dim=1)`，兩分支輸出在分類頭之前才拼接
- **遷移學習**：載入ImageNet預訓練權重
- **動態維度適配**：`img_feat_dim = self.img_encoder.num_features` 自動抓取，換backbone不用改其他程式碼
- **凍結/解凍機制**：訓練初期凍結backbone，只訓練新模組，穩定後才解凍一起微調

### Block L — Loss Function

- **類別加權交叉熵**：讓稀有類別的錯誤在loss計算時被放大
- **權重截斷**：避免極端不平衡算出過大權重
- **Focal Loss（選用）**：讓模型聚焦在難分類樣本上

### Block M — Optimizer

- **分層學習率**：預訓練backbone用較小LR（微調），新模組用較大LR（從頭學）
- **AdamW**：解耦權重衰減，比傳統Adam+L2更穩定

### Block N — LR Scheduler

- 可透過 `SCHEDULER_TYPE` 切換 `"plateau"`（ReduceLROnPlateau，表現驅動）或 `"cosine"`（CosineAnnealingLR，時間驅動）
- **`step_scheduler()`統一入口**：依`SCHEDULER_TYPE`決定`step()`要不要傳指標值，Block R的`main()`只需呼叫這個函式，之後切換scheduler種類不用改主流程
- **注意**：若用plateau模式，scheduler的`patience`要明顯小於`EARLY_STOP_PATIENCE`，否則兩者同時觸發，LR才剛降訓練就被停止，scheduler等於沒發揮作用

### Block O — Mixup（預設關閉）

特徵空間線性插值增強，本專案因ECG印刷影像的像素混合可能產生不具臨床意義的波形疊加，預設關閉。人口學特徵不參與混合。

### Block P — 訓練與驗證迴圈

- `train_one_epoch()` 同步計算 **train accuracy**（不只算loss），跟`evaluate()`回傳的val accuracy一起呈現在訓練log裡，方便直接比較train/val的落差判斷過擬合程度
- `torch.no_grad()` 停用梯度計算，節省驗證階段資源
- **Macro average 指標**：precision/recall/f1都採macro平均，更真實反映稀有類別表現
- **`VERBOSE_BATCH`開關**：預設關閉，只有需要逐batch除錯時才開啟，正式訓練時log保持簡潔（一個epoch一行摘要）

### Block Q — Grad-CAM（可解釋性）

- 透過forward/backward hook抓取目標層的activation與梯度，計算出反映「模型關注哪些影像區域」的熱力圖
- **視覺化輸出**：`plot_gradcam_single()` 產出原圖/純熱力圖/疊圖三合一，`generate_gradcam_examples()` 每個類別各挑樣本自動產圖
- **侷限**：僅適用於CNN空間特徵，只能解釋影像分支貢獻，人口學分支無法用此方式視覺化

### Block R — 主流程（`main()`）

- **端到端pipeline編排**：資料處理→模型建構→訓練→best checkpoint回載→test評估→自動產圖
- **Best checkpoint選擇**：追蹤驗證集表現最好的epoch權重，而非用最後一個epoch
- **Early stopping**：連續`EARLY_STOP_PATIENCE`個epoch沒進步就提前終止
- **結構化訓練Log**：每個epoch輸出一行摘要（含backbone名稱、train/val loss、train/val accuracy、precision/recall/f1、耗時），並在backbone解凍瞬間、best checkpoint更新、early stopping倒數時額外印出標記訊息，方便長時間訓練時快速掃視進度，範例：

  ```
  開始訓練 convnext_tiny
  =======================================================
    Backbone 已凍結，僅訓練分類頭
  [convnext_tiny] Epoch 01/80 | Loss: 1.4150/1.2044 | Acc: 0.3330/0.4820 | Prec: 0.4461 | Rec: 0.4357 | F1: 0.3572 | Time: 31.6s
    → ⭐ 已儲存最佳模型（F1: 0.3572）

  🔥 [convnext_tiny] Epoch 6：解凍 backbone
    分層學習率：分類頭 1e-4 / backbone 1e-5
  [convnext_tiny] Epoch 06/80 | Loss: 1.2941/1.0708 | Acc: 0.4277/0.5899 | Prec: 0.7370 | Rec: 0.6242 | F1: 0.5512 | Time: 26.9s
    → 未進步（1/20）
  ```

### Block S — 視覺化

- **YOLO風格六宮格總覽**（`00_results_grid.png`）、**混淆矩陣雙版本**（`02_confusion_matrix.png`）、**各類別指標長條圖**（`03_per_class_metrics.png`）、**不平衡對照圖**（`04_imbalance_vs_recall.png`）、**整體彙總表**（`05_summary_table.png`/`.csv`）
- **`print_overall_summary()`**：訓練結束印出簡潔文字版整體結果，方便快速複製紀錄

---

## 5. 執行方式

1. 依序執行 Block A → S
2. （建議）跑完Block D後，先用 `visualize_augmentations()` 確認資料增強效果合理
3. 執行 `model, test_metrics = main(CONFIG)`
4. 建議用 **Save Version → Save & Run All (Commit)** 執行，避免 session 過期導致輸出被清空

**⚠️ 常見疏漏**：不要在最後一個cell重複呼叫 `main(CONFIG)` 兩次，會讓整個訓練流程跑兩遍，白白浪費運算時間。

輸出檔案：`00_results_grid.png`、`01_training_curves.png`、`02_confusion_matrix.png`、`03_per_class_metrics.png`、`04_imbalance_vs_recall.png`、`05_summary_table.png/csv`、`augmentation_preview.png`、`gradcam_{class}_0.png` × 5、`best_model.pth`

---

## 6. 已知結果與問題

- **年齡模態 ablation**：初期版本對5類別分類任務貢獻約±0.01 F1，效果不顯著，晚期融合強度可能是限制因素
- **Backbone對照（同一套正則化配方）**：ConvNeXt-atto的test表現目前優於ResNet18（macro F1約0.66 vs 0.58），但訓練曲線的過擬合傾向也更明顯，需要更謹慎監控best checkpoint的選取時機；ResNet18訓練曲線最穩定、風險最低，但表現上限目前看來較低
- **NORM/HYP的precision-recall trade-off**：WeightedRandomSampler讓稀有類別recall提升的同時，也讓多數類別容易被誤判

---

## 7. 待辦事項

- [ ] Checkpoint / Resume 機制（尚未整合進本檔案）
- [ ] ConvNeXt專屬正則化：`drop_path_rate`（Stochastic Depth），目前尚未加入Block K
- [ ] 年齡分組 embedding 或 FiLM 融合，取代目前的 scalar + concat 方式
- [ ] 遷移至 KD (Kawasaki Disease) ECG 資料集（ZZU pECG dataset）
