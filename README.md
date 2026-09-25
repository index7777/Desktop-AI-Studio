# Desktop AI Studio

Desktop AI Studio 是一個以本機 AI 生圖為核心的桌面應用專案。

## 產品方向

- 自有乾淨 UI，不暴露 ComfyUI 節點
- 優先採用本機推理架構
- 目標支援 Qwen Image 系列
- 規劃加入 Reference Image、圖片編輯、Pose / 骨架、模型管理、硬體偵測與生成紀錄
- 桌面端預計採用 React + TypeScript + Tauri
- AI Engine 預計採用 Python + PyTorch + Diffusers

## UI Master

目前已確認一版深色桌面應用母版，核心區域包含：

1. 生成
2. 圖片編輯
3. 姿勢 / 骨架
4. 模型管理
5. 設定

後續實作將以此母版為 Design Baseline，避免重新發散整體介面架構。

## 第一階段目標

完成可執行的 UI Shell：

- 左側主導覽
- Prompt 輸入
- 參考圖區
- Pose 入口
- 尺寸比例選擇
- Advanced Settings
- 中央圖片預覽
- 右側生成歷史
- 模型管理頁
- 設定頁

第二階段再接入本機 Qwen Image 推理流程。
