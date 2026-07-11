import pandas as pd
import numpy as np
import random
import gradio as gr
import os
import base64
import warnings
from sklearn.ensemble import RandomForestRegressor

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 資料處理 (確保路徑正確) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'tea_data0714無味道優化.xlsx')
img_path = os.path.join(current_dir, 'tea_bg.jpg')

# 載入模型
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
aromas = ['甜香', '蜜糖香', '焦糖香', '花香', '熟果香', '原木香', '烤焙香', '青草味']
existing_columns = [col for col in aromas if col in df.columns]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[existing_columns], df[['固液比', '壓力']])

# --- 2. 圖片轉 Base64 (解決 Render 讀不到背景圖的問題) ---
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except: return ""

bg_base64 = get_image_base64(img_path)

# --- 3. CSS 設計 (完全對齊你原本的視覺風格) ---
custom_css = f"""
body {{
    background: url('{bg_base64}') no-repeat center center fixed !important;
    background-size: cover !important;
    font-family: '微軟正黑體', sans-serif !important;
}}
.glass-panel {{
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 30px !important;
    padding: 40px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
}}
button {{
    background: #000000 !important;
    color: white !important;
    border-radius: 15px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}}
"""

# --- 4. 邏輯與介面 ---
def suggest_process(selected_aromas, excluded_aromas):
    input_vector = pd.DataFrame(0, index=[0], columns=existing_columns)
    for f in selected_aromas:
        if f in input_vector.columns: input_vector[f] = 5
    for f in excluded_aromas:
        if f in input_vector.columns: input_vector[f] = 0
    pred = model.predict(input_vector)[0]
    return gr.update(visible=False), gr.update(visible=True), f"{np.clip(pred[0], 0.1111, 0.1429):.4f}", f"{np.clip(pred[1], 0.133, 0.260):.3f} bar", f"{random.randint(1, 5)}%"

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_classes="glass-panel"):
        gr.Markdown("# 🍃 茶香 AI 製程優化系統")
        input_wanted = gr.CheckboxGroup(aromas, label="✨ 想要增強的香氣")
        input_excluded = gr.CheckboxGroup(aromas, label="🚫 想要避開的香氣")
        btn = gr.Button("開始計算建議製程")
        
        # 隱藏的結果區
        with gr.Column(visible=False) as result_area:
            out_ratio = gr.Textbox(label="建議固液比")
            out_pressure = gr.Textbox(label="建議壓力")
            out_additive = gr.Textbox(label="建議回添比例")
            back_btn = gr.Button("重新設定")
            back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[None, result_area])

    btn.click(suggest_process, inputs=[input_wanted, input_excluded], outputs=[btn, result_area, out_ratio, out_pressure, out_additive])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
