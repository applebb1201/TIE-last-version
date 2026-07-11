import pandas as pd
import numpy as np
import random
import gradio as gr
import time
import base64
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- 設定路徑 ---
# 自動獲取程式碼所在目錄，確保雲端也能讀到檔案
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / 'tea_data0714無味道優化.xlsx'
img_path = BASE_DIR / "tea_bg.jpg"

# --- 1. 資料處理 ---
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

aromas = ['甜香', '蜜糖香', '焦糖香', '花香', '熟果香', '原木香', '烤焙香', '青草味']
existing_columns = [col for col in aromas if col in df.columns]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[existing_columns], df[['固液比', '壓力']])

# --- 2. 圖片處理 ---
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

base64_bg = get_image_base64(img_path)

# --- 3. 美學 CSS (維持不變) ---
custom_css = f"""
.gradio-container {{ background: url('{base64_bg}') no-repeat center center fixed !important; background-size: cover !important; font-family: sans-serif !important; }}
.glass-panel {{ background: rgba(255, 255, 255, 0.85) !important; backdrop-filter: blur(25px) !important; border-radius: 40px !important; padding: 50px !important; }}
"""

# --- 4. 邏輯函數 ---
def suggest_process(selected_aromas, excluded_aromas):
    time.sleep(0.5)
    input_vector = pd.DataFrame(0, index=[0], columns=existing_columns)
    for f in selected_aromas:
        if f in input_vector.columns: input_vector[f] = 5
    for f in excluded_aromas:
        if f in input_vector.columns: input_vector[f] = 0
    pred = model.predict(input_vector)[0]
    return gr.update(visible=False), gr.update(visible=True), f"{np.clip(pred[0], 0.1111, 0.1429):.4f}", f"{np.clip(pred[1], 0.133, 0.260):.3f} bar", f"{random.randint(1, 5)}%"

# --- 5. 介面 ---
with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_classes="glass-panel"):
        with gr.Column(visible=True) as main_area:
            gr.Markdown("# 🍃 茶香 AI 製程優化系統")
            input_wanted = gr.CheckboxGroup(aromas, label="✨ 想要增強的香氣")
            input_excluded = gr.CheckboxGroup(aromas, label="🚫 想要避開的香氣")
            btn = gr.Button("開始計算建議製程")

        with gr.Column(visible=False) as result_area:
            gr.Markdown("## 📋 建議製程參數")
            out_ratio = gr.Textbox(label="建議固液比")
            out_pressure = gr.Textbox(label="建議壓力")
            out_additive = gr.Textbox(label="建議回添比例")
            back_btn = gr.Button("重新設定")
            back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[main_area, result_area])

    btn.click(suggest_process, inputs=[input_wanted, input_excluded], outputs=[main_area, result_area, out_ratio, out_pressure, out_additive])

# --- 關鍵修改：讓雲端平台指定連接埠 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
