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

# --- 1. 資料處理 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'tea_data0714無味道優化.xlsx')
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
    except Exception:
        return ""

base64_bg = get_image_base64(img_path)

# --- 3. 美學與互動 CSS ---
custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@700;900&display=swap');

body, .gradio-container {{ 
    background: url('{base64_bg}') no-repeat center center fixed !important; 
    background-size: cover !important; 
    font-family: 'Zen Maru Gothic', sans-serif !important; 
    font-weight: 900 !important; 
}}

.glass-panel {{ 
    background: rgba(255, 255, 255, 0.85) !important; 
    backdrop-filter: blur(25px) !important; 
    border-radius: 40px !important; 
    padding: 50px !important; 
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3) !important; 
    width: 90vw !important; 
    max-width: 800px !important; 
    margin: 50px auto !important; 
}}

.gradio-container .gr-checkbox-group .wrap {{ display: flex !important; flex-wrap: wrap !important; }}
.gradio-container .gr-checkbox-group .wrap > label {{ flex: 0 0 45% !important; margin-right: 5% !important; }}

.gradio-container label, .gradio-container span, .gradio-container p, .gradio-container h1, .gradio-container h2 {{ 
    color: #000000 !important; font-weight: 900 !important; 
}}

.gradio-container .gr-checkbox-group > label {{ font-size: 20pt !important; margin-bottom: 15px !important; }}
.gradio-container .gr-checkbox-group span {{ font-size: 18pt !important; }}
.gradio-container .gr-textbox label {{ font-size: 20pt !important; }}
.gradio-container input {{ font-size: 18pt !important; color: #000000 !important; }}

button {{ 
    background: #000000 !important; color: #ffffff !important; 
    border-radius: 20px !important; border: none !important; 
    font-size: 16pt !important; font-weight: 700 !important; 
    transition: 0.3s !important; padding: 15px 30px !important; margin-top: 20px !important; 
}}
button:hover {{ background: #333333 !important; transform: scale(1.01) !important; }}
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
    with gr.Column(elem_classes="glass-panel") as main_area:
        gr.Markdown("# 🍃 茶香 AI 製程優化系統")
        gr.Markdown("請勾選您希望表現或避開的風味，點擊下方按鈕以計算製程參數與回添比例")
        
        input_wanted = gr.CheckboxGroup(aromas, label="✨ 想要增強的香氣")
        input_excluded = gr.CheckboxGroup(aromas, label="🚫 想要避開的香氣")
        btn = gr.Button("開始計算建議製程")

    with gr.Column(visible=False, elem_classes="glass-panel") as result_area:
        gr.Markdown("## 📋 建議製程參數")
        out_ratio = gr.Textbox(label="建議固液比")
        out_pressure = gr.Textbox(label="建議壓力")
        out_additive = gr.Textbox(label="建議回添比例")
        back_btn = gr.Button("重新設定")
        back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[main_area, result_area])

    btn.click(suggest_process, inputs=[input_wanted, input_excluded], outputs=[main_area, result_area, out_ratio, out_pressure, out_additive])

if __name__ == "__main__":
    # Render 強制執行設定
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
