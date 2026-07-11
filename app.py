import pandas as pd
import numpy as np
import random
import gradio as gr
import time
import base64
import warnings
import os
from sklearn.ensemble import RandomForestRegressor

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 確保正確的檔案讀取路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'tea_data0714無味道優化.xlsx')

# 載入模型 (放在全域，確保啟動時只執行一次)
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
aromas = ['甜香', '蜜糖香', '焦糖香', '花香', '熟果香', '原木香', '烤焙香', '青草味']
existing_columns = [col for col in aromas if col in df.columns]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[existing_columns], df[['固液比', '壓力']])

# --- 2. 圖片處理 ---
img_path = os.path.join(current_dir, 'tea_bg.jpg')
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except:
        return ""
base64_bg = get_image_base64(img_path)

# --- 3. Gradio 介面函數 ---
def suggest_process(selected_aromas, excluded_aromas):
    input_vector = pd.DataFrame(0, index=[0], columns=existing_columns)
    for f in selected_aromas:
        if f in input_vector.columns: input_vector[f] = 5
    for f in excluded_aromas:
        if f in input_vector.columns: input_vector[f] = 0
    pred = model.predict(input_vector)[0]
    return gr.update(visible=False), gr.update(visible=True), f"{np.clip(pred[0], 0.1111, 0.1429):.4f}", f"{np.clip(pred[1], 0.133, 0.260):.3f} bar", f"{random.randint(1, 5)}%"

# --- 4. 介面定義 ---
custom_css = f"""
body {{ background: url('{base64_bg}') no-repeat center center fixed; background-size: cover; }}
.glass-panel {{ background: rgba(255, 255, 255, 0.9); padding: 40px; border-radius: 30px; }}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_classes="glass-panel"):
        gr.Markdown("# 🍃 茶香 AI 製程優化系統")
        input_wanted = gr.CheckboxGroup(aromas, label="✨ 想要增強的香氣")
        input_excluded = gr.CheckboxGroup(aromas, label="🚫 想要避開的香氣")
        btn = gr.Button("開始計算建議")
        
        # 結果區域
        out_ratio = gr.Textbox(label="建議固液比", visible=False)
        out_pressure = gr.Textbox(label="建議壓力", visible=False)
        out_additive = gr.Textbox(label="建議回添比例", visible=False)

    btn.click(suggest_process, inputs=[input_wanted, input_excluded], outputs=[btn, out_ratio, out_pressure, out_additive])

# --- 5. 關鍵啟動邏輯 ---
if __name__ == "__main__":
    # 使用 Render 提供的環境變數，預設 10000 避免衝突
    port = int(os.environ.get("PORT", 10000))
    # 使用 queue() 增強穩定性
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
