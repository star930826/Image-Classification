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

---

## 3. 逐 Block 技術詳解

### Block A — 全域設定（CONFIG）

所有超參數與開關集中管理，是整個 pipeline 唯一需要修改的地方（正常使用情境下）。技術重點：

- **關注點分離（separation of concerns）**：路徑、任務設定、模型架構、訓練超參數、不平衡處理、資料增強、Grad-CAM、early stopping 分區塊管理，避免參數散落在程式碼各處難以維護
- **Ablation 開關設計**：`USE_AGE_MODALITY`、`USE_SEX_MODALITY` 等旗標讓同一份程式碼可以做「有/無某個模態」的對照實驗，不用複製整份程式碼維護兩個版本

### Block B — 隨機種子固定

```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

固定 Python、NumPy、PyTorch（CPU + GPU）四處不同的隨機數來源，並關閉 cuDNN 的自動優化搜尋（`benchmark=False`）換取可重現性（但會犧牲一點訓練速度）。這是**實驗可重現性（reproducibility）**的標準做法——沒有這一步，同樣的 CONFIG 兩次執行結果可能不同，會讓 ablation 比較失去意義。

### Block C — Metadata 讀取與診斷聚合

技術重點：

- `ast.literal_eval`：PTB-XL 的 `scp_codes` 欄位在 CSV 裡是字串形式的 Python dict（如 `"{'NORM': 100.0}"`），用這個函式安全地把字串解析回真正的 dict 物件（比 `eval()` 安全，只允許解析基本資料型態，不會執行任意程式碼）
- **診斷代碼聚合（code aggregation）**：把細部診斷代碼（`scp_codes`，上百種）透過 `scp_statements.csv` 的對照表，映射到 5 個 superclass 大類，這是醫學影像/訊號分類任務常見的**標籤層次化（label hierarchy）**處理方式
- **多標籤過濾為單標籤**：`diagnostic_superclass` 長度 >1 的紀錄（同時符合多個大類）被濾除，簡化成標準的 multi-class（而非 multi-label）分類問題

### Block D — 年齡/性別清理 + 影像路徑展開

技術重點：

- **缺值處理（missing value handling）**：`clean_age()`、`clean_sex()` 都採用「直接濾除缺值紀錄」的策略（而非插補 imputation），因為年齡/性別是本任務的必要輸入模態，插補容易引入偏差，且缺值比例低（約38筆/16000+筆），直接丟棄成本很低
- **去識別化資料的邊界值處理**：PTB-XL 對 >89 歲的紀錄依 HIPAA 規範統一設為 300 歲，`clip(upper=clip_max)` 把這類邊界值裁切回合理範圍，避免離群值扭曲後續 z-score 標準化的 mean/std
- **確定性路徑映射（deterministic path construction）**：`get_ptbxl_folder()` 用整數除法 `(ecg_id // 1000) * 1000` 計算資料夾分組，是 PTB-XL 官方資料集的標準佈局規則，避免單一資料夾檔案數過多
- **長表格式（long format）轉換**：`build_image_index()` 把「一筆病人紀錄」展開成「一筆影像」的長表，是多模態資料整併的常見前處理型態，方便後續 Dataset 直接逐行讀取

### Block E — Patient-level 資料切分

技術重點：

- **`GroupShuffleSplit`（分組切分）**：一般的 `train_test_split` 是逐筆隨機切分，但同一病人（`patient_id`）可能有多筆 ECG 紀錄，如果隨機切分，同一人的資料可能同時出現在 train 和 test，造成**資料洩漏（data leakage）**——模型等於「看過」測試病人的其他心電圖，測出來的表現會虛高。`GroupShuffleSplit` 以 `patient_id` 為切分單位，保證同一病人的所有紀錄都在同一個 split
- **兩階段切分**：先切出 test set，再從剩餘資料切出 val set（`relative_val_size` 換算），確保三個 split 的病人完全互斥（程式碼裡有 `assert` 硬性檢查交集為空集合）

### Block F — 年齡 Z-score 標準化

```python
z = (x - mean) / std
```

技術重點：

- **特徵標準化（feature standardization）**：把原始年齡（範圍約2–90）轉換成均值0、標準差1的分布，避免數值尺度差異過大的特徵在訓練初期主導梯度更新方向
- **避免資料洩漏的標準化流程**：`AgeScaler.fit()` 只用 train set 計算 mean/std，val/test set 只呼叫 `transform()` 套用已經算好的統計量——這是機器學習 pipeline 的標準做法，用全體資料算統計量等於讓模型間接看到了測試集的分布資訊

### Block G — 影像資料增強

技術重點：

- **保守的增強策略**：只用 `RandomRotation(3)`（極小角度旋轉）和 `ColorJitter`（亮度/對比度微調），刻意不用會破壞醫學影像語義的增強（如水平/垂直翻轉、大角度旋轉），因為 ECG 波形的時間軸方向和振幅方向都帶有臨床意義，翻轉會產生不存在的病理特徵
- **ImageNet 標準化**：`Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` 是 ImageNet 預訓練模型的標準輸入分布，因為 `img_encoder` 用的是 ImageNet 預訓練權重（遷移學習），輸入分布需要跟預訓練時一致，模型才能有效利用預訓練特徵

### Block H — 多模態 Dataset

技術重點：

- **多模態資料封裝**：`__getitem__` 一次回傳三個張量（image, demo, label），這是 PyTorch 多模態任務的標準 Dataset 設計模式——把不同模態的讀取、前處理邏輯封裝在同一個介面下，DataLoader 可以透明地批次組合
- **動態特徵組合**：`demo_values` 用 list 動態組裝，依 `use_age`/`use_sex` 開關決定最終 `demo_tensor` 的維度（1維或2維），讓同一個 Dataset class 同時支援單模態/雙模態組合，不用為每種組合寫不同的 class
- **類別特徵的處理方式差異**：性別（二元類別）不做 z-score，直接當數值輸入（0/1），因為 z-score 是為了處理有意義的連續尺度差異，二元類別本身沒有「尺度」概念

### Block I — WeightedRandomSampler

技術重點：

- **逆頻率加權抽樣（inverse frequency weighted sampling）**：`weight = total_samples / class_count`，樣本數越少的類別，被抽中的機率權重越高。這是處理類別不平衡最常見的技術之一，讓模型在訓練時看到各類別的**有效頻率**趨近於平衡，而不是被多數類別（NORM）主導梯度更新方向
- **`replacement=True`**：允許同一筆資料在一個 epoch 內被重複抽樣，這是配合權重抽樣的必要設定（稀有類別樣本數不夠，要抽出跟多數類別一樣多的樣本次數必須靠重複抽樣達成）

### Block J — DataLoader 建構

技術重點：

- **train/val/test 三個 DataLoader 分流處理**：只有 train_loader 套用 `WeightedRandomSampler`，val/test 維持原始分布（`shuffle=False`）——因為驗證/測試階段要評估模型在**真實世界分布**下的表現，不應該用重新加權後的分布去評估，否則算出來的指標無法反映實際部署時的表現

### Block K — 多模態融合模型

技術重點：

- **雙分支編碼器架構（two-branch encoder architecture）**：影像走 CNN backbone（`timm.create_model`，支援任意 timm 支援的架構），人口學特徵走小型 MLP，兩者分別抽取各自模態的高層次特徵
- **晚期融合（late fusion）**：`torch.cat([img_feat, demo_feat], dim=1)`，兩個分支的輸出特徵在**分類頭之前**才拼接，是多模態任務中最穩定、最容易訓練的融合策略（相對於在輸入層就合併的 early fusion）
- **遷移學習（transfer learning）**：`timm.create_model(backbone, pretrained=True, num_classes=0)` 載入 ImageNet 預訓練權重，`num_classes=0` 拿掉原本的分類層，只保留特徵抽取部分（feature extractor）
- **動態維度適配**：`img_feat_dim = self.img_encoder.num_features` 自動讀取當前 backbone 的輸出維度，換 backbone（如 ResNet18→ConvNeXt）時不需要手動調整後續層的輸入維度
- **凍結/解凍機制（freeze/unfreeze，漸進式微調 progressive fine-tuning）**：`freeze_backbone()`/`unfreeze_backbone()` 控制 `img_encoder` 參數是否參與梯度更新，訓練初期凍結預訓練權重、只訓練新加的分類頭與人口學分支，讓新模組先穩定下來，再解凍整個網路一起微調，避免預訓練權重在訓練初期被不成熟的梯度訊號破壞

### Block L — Loss Function

技術重點：

- **類別加權交叉熵（class-weighted cross entropy）**：`weight = total_samples / class_count`，讓稀有類別的錯誤在 loss 計算時被放大，跟 WeightedRandomSampler 是處理類別不平衡的兩種不同機制（一個作用在抽樣階段，一個作用在損失計算階段），本專案同時採用兩者
- **權重截斷（weight clipping）**：`torch.clamp(weights, min=0.5, max=5.0)` 避免極端不平衡（如 HYP 樣本數是 NORM 的1/17）算出過大的權重，導致模型過度偏向稀有類別、犧牲多數類別的表現
- **Focal Loss（選用）**：`(1-pt)^gamma * ce_loss`，讓模型「容易分類」的樣本（pt 高、loss 低）貢獻更小的梯度，聚焦在「難分類」的樣本上，是目標檢測/不平衡分類任務常用的 loss 設計，本專案作為 class-weighted CE 的替代方案

### Block M — Optimizer

技術重點：

- **分層學習率（discriminative learning rate / layer-wise LR）**：`img_encoder`（預訓練 backbone）用較小的 `LR_BACKBONE`，`classifier`+`demo_encoder`（從頭訓練的新模組）用較大的 `LR_HEAD`——預訓練權重已經在大型資料集上收斂過，只需要小幅微調；新模組是隨機初始化，需要較大的學習率才能快速學到有用的特徵
- **AdamW**：Adam 優化器搭配解耦的權重衰減（decoupled weight decay），比傳統 Adam + L2 正則化在正則化效果上更穩定，是目前 CNN/Transformer 訓練的主流選擇

### Block N — LR Scheduler

技術重點：

- **`ReduceLROnPlateau`（表現驅動的學習率衰減）**：監控 `val_f1_macro`，連續 `patience=5` 個 epoch 沒有進步就把 LR 減半（`factor=0.5`），直到 `min_lr=1e-6` 為止。相對於 `CosineAnnealingLR` 這種依固定 epoch 數排程衰減的方式，`ReduceLROnPlateau` 是**根據模型實際表現**動態決定何時降低學習率，更符合「讓模型充分收斂」的目標——模型還在進步時不會被提前降速，卡住時才介入

### Block O — Mixup（預設關閉）

技術重點：

- **Mixup 資料增強**：`mixed_image = λ*image_A + (1-λ)*image_B`，並用 Beta 分布抽樣混合係數 λ，是一種在**特徵空間**做線性插值的正則化技術，能有效抑制過擬合、提升模型泛化能力，常用於自然影像分類。本專案預設關閉，因為 ECG 印刷影像的像素混合可能產生不具真實臨床意義的波形疊加，需要審慎評估後再啟用
- **人口學特徵不參與混合**：`demo` 維持原值傳遞（不像影像做線性插值），避免年齡/性別這種離散/類別特徵被「混合」成沒有意義的中間值（例如兩個性別的 0.5/0.5 混合沒有臨床意義）

### Block P — 訓練與驗證迴圈

技術重點：

- **`torch.no_grad()` 裝飾器**：`evaluate()` 函式停用梯度計算，節省驗證階段的記憶體與運算時間（驗證不需要反向傳播）
- **Macro average 指標**：`f1_score(..., average="macro")`、`precision_score`、`recall_score` 都採 macro 平均——先分別計算每個類別的指標，再取算術平均（每個類別權重相同），跟 accuracy 或 weighted average 相比，能更真實反映模型在**稀有類別**上的表現，不會被多數類別（NORM）的高表現掩蓋少數類別的低表現
- **Batch-level 進度顯示**：`train_one_epoch()` 每 50 個 batch 印一次 loss，方便在長時間訓練中即時確認程式是否正常執行（不是卡住），也能及早發現 loss 爆炸/NaN 等異常

### Block Q — Grad-CAM（可解釋性）

技術重點：

- **Grad-CAM（Gradient-weighted Class Activation Mapping）**：透過 forward hook 抓取目標卷積層的 activation（`_save_activation`），backward hook 抓取該層的梯度（`_save_gradient`），用梯度對每個 channel 做全域平均池化當作權重（`weights = gradients.mean(dim=(2,3))`），加權加總 activation 後過 ReLU，得到一張反映「模型在做這個分類決策時，影像哪些區域最重要」的熱力圖
- **僅適用於 CNN 空間特徵**：這個實作依賴 activation 是 `(batch, channel, height, width)` 的空間特徵圖格式，只適用於 CNN backbone；若未來換成 ViT/Swin 等 Transformer 架構，中間層輸出是序列格式，需要改用其他可解釋性方法（如 Attention Rollout）
- **多模態任務下的可解釋性侷限**：Grad-CAM 只能解釋影像分支的貢獻，人口學分支（scalar 輸入）沒有空間結構，無法用同樣方式視覺化，解讀時需要註明這個侷限

### Block R — 主流程（`main()`）

技術重點：

- **端到端 pipeline 編排**：依序呼叫資料處理 → 模型建構 → 訓練迴圈 → 最佳權重回載 → test set 評估 → 自動產圖 → Grad-CAM 範例，是完整的機器學習實驗流程封裝
- **Best checkpoint 選擇（而非用最後一個 epoch）**：`if current_metric > best_metric: best_state = copy.deepcopy(model.state_dict())`，訓練過程中持續追蹤驗證集表現最好的那個 epoch 權重，最終用這組權重跑 test set，而不是用訓練結束時的最後權重——因為最後幾個 epoch 可能已經過擬合，用 best checkpoint 是標準做法
- **Early stopping**：`patience_counter` 累計「沒有超越目前最佳表現」的連續 epoch 數，達到 `EARLY_STOP_PATIENCE` 就提前終止訓練，避免無意義地繼續訓練已經收斂（或退化）的模型，節省運算資源

### Block S — 視覺化

技術重點：

- **CJK 字型動態載入**：`setup_cjk_font()` 偵測系統是否有 Noto Sans CJK 字型並註冊給 matplotlib，避免中文標籤顯示成方框（tofu）警告；本專案圖表實際採用英文標籤以確保跨環境穩定性
- **訓練歷程視覺化（YOLO 風格六宮格）**：`plot_yolo_style_results()` 同時呈現 train/val loss、precision/recall/f1/accuracy 隨 epoch 的走勢，並疊加移動平均平滑線（`np.convolve` 滑動窗口平均），標註 backbone 解凍點與 best checkpoint 位置，方便一眼判斷訓練是否穩定收斂
- **混淆矩陣雙版本呈現**：數量版 + 列正規化百分比版並列，數量版反映絕對規模，百分比版方便跨類別比較 recall 表現（因為每個類別樣本數差異很大，只看數量容易誤判）
- **Support vs Recall 對照圖**：雙 Y 軸疊圖，直接視覺化驗證「類別不平衡處理策略是否真的影響了各類別的召回率」，是判斷不平衡處理是否過度/不足的診斷工具

---

## 4. 執行方式

1. 依序執行 Block A → S
2. 執行 `model, test_metrics = main(CONFIG)`
3. 建議用 **Save Version → Save & Run All (Commit)** 執行，避免 session 過期導致 `/kaggle/working` 輸出被清空

輸出檔案：`00_results_grid.png`、`01_training_curves.png`、`02_confusion_matrix.png`、`03_per_class_metrics.png`、`04_imbalance_vs_recall.png`、`best_model.pth`、`sample_gradcam.npy`

---

## 5. 目前已知結果與問題

- **年齡模態 ablation**：對 5 類別分類任務貢獻約在 ±0.01 F1，落在訓練隨機性範圍內，效果不顯著
- **NORM 類別 recall 偏低（0.55–0.57）**：WeightedRandomSampler 拉高稀有類別抽樣頻率的副作用
- **HYP 類別 precision 偏低（0.33–0.37）**：同上，稀有類別 recall 提升的 trade-off

---

## 6. 待辦事項

- [ ] Checkpoint / Resume 機制（尚未整合進本檔案，僅完成設計討論）
- [ ] 年齡分組 embedding 或 FiLM 融合，取代目前的 scalar + concat 方式
- [ ] 遷移至 KD (Kawasaki Disease) ECG 資料集（ZZU pECG dataset）
