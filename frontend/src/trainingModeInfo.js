// Explanations describe the modes implemented by this workbench, not a promise
// that every model family expands its classifier in the same way.
export const trainingModeInfo = {
  standard: {
    title: '標準微調', shortTitle: '標準微調',
    summary: '把模型調整成這次資料集的任務，輸出使用資料集定義的類別。',
    data: '只需新任務的訓練與驗證資料。',
    updates: '依下方微調策略，更新全部或部分參數。',
    retention: '資料集沒有列出的原類別，不會自動保留。這不是在原有類別後面追加。',
    inference: '一個模型、一次推論；輸出這次資料集的類別。',
    useCase: '只需要辨識便當菜色，不需要繼續輸出原本的 COCO 類別。',
  },
  expand_frozen: {
    title: '凍結骨幹後擴充', shortTitle: '凍結擴充',
    summary: '固定特徵骨幹，讓同一個模型學習新增類別，保留新舊類別的輸出範圍。',
    data: '新資料即可。舊資料選填，提供時只用驗證切分量測舊能力，不混入訓練。',
    updates: '偵測模型固定骨幹，其他偵測模組仍可更新；分類模型只訓練分類頭。舊分類權重也可能更新，不是只訓練新增類別。',
    retention: '不保證舊能力不下降。骨幹不變，不等於原本的判斷不變；未提供舊驗證資料時，舊能力尚未測量。',
    inference: '一個模型、一次推論；不需另外跑原模型。',
    useCase: '沒有舊訓練資料，想維持單一模型，且能接受舊類別效果需要再驗證。',
  },
  expand_classes: {
    title: '混合新舊資料擴充', shortTitle: '混合資料',
    summary: '讓同一個模型同時練習新任務與舊任務，降低只學新資料造成的遺忘。',
    data: '必須提供新資料，以及涵蓋原有類別的舊訓練與驗證資料。',
    updates: '擴充新舊類別的輸出範圍，混合兩份訓練資料；更新哪些參數仍依下方微調策略。',
    retention: '舊資料能幫助保留能力，但仍可能退步；工作台會比較舊任務訓練前後的驗證指標。',
    inference: '一個模型、一次推論；訓練資料量增加，訓練通常更花時間。',
    useCase: '手上有原任務資料，想用一個模型同時辨識原有物件與便當菜色。',
  },
  dual_branch: {
    title: '固定原模型＋新增分支', shortTitle: '固定＋分支',
    summary: '原模型完整保留，另外訓練一個新模型分支；不是在共用骨幹上只加一個小分類頭。',
    data: '新資料即可。舊資料選填，提供時只用驗證切分評估原分支。',
    updates: '原分支不參與訓練；新分支獨立訓練，更新範圍依下方微調策略。',
    retention: '相同輸入與參數下保留原分支的預測。新增分支仍可能判錯，兩分支可能有重複或矛盾的框；分數不能當成共同機率。',
    inference: '依序跑兩個分支，權重儲存與推論耗時增加。目前支援圖片／影片比較，即時攝影機尚未接入。',
    useCase: '優先保留原模型的結果，並能接受額外的模型容量與推論時間。',
  },
};
