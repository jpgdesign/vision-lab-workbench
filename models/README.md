# 本機模型目錄

可以把預訓練權重放在這裡，或在工作台的「模型與任務」登錄既有檔案／資料夾的絕對路徑。
登錄外部路徑不會搬移或複製權重。請保留 Hugging Face 模型的設定、處理器及 tokenizer 等必要檔案。

此目錄中的權重不會加入 Git。支援的模型及微調格式見專案 [README](../README.md)。

各預設版本的官方下載連結、大小與可複製指令，請見 [預設模型下載表](../README.md#預設模型下載)。

應用工作檯的額外骨架／關係模型可在專案根目錄執行 `python download_workflow_models.py` 查看計畫，再加 `--download --models pose skeleton relation` 下載。預設建立 `yolo26n_pose/`、`stgcnpp_ntu120/`、`relateanything_convnext_large/` 三個子目錄；`--verify` 可核對大小與 SHA256。模型檔與設定應完整保留，授權、容量及其他搜尋位置見 [多模型應用工作檯](../README.md#多模型應用工作檯)。
