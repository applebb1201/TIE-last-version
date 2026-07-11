import pandas as pd
import numpy as np
import random
import gradio as gr
import os
import base64
from sklearn.ensemble import RandomForestRegressor

# --- 強制設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'tea_data0714無味道優化.xlsx')
img_path = os.path.join(current_dir, 'tea_bg.jpg')

# --- 載入模型 ---
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
aromas = ['甜香', '蜜糖香', '焦糖香', '花香', '熟果香', '原木香', '烤焙香', '青草味']
existing_columns = [col for col in aromas if col in df.columns]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[existing_columns], df[['固液比', '壓力']])

# --- 圖片轉碼 (解決背景消失) ---
def get_image_base64(path):
    try:
        with open(path, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    except: return ""

bg_base64 = get_image_base64(img_path)

# --- CSS (強制修正排版) ---
custom_css = f"""
body {{ background: url('{bg_base64}') no-repeat center center fixed !important; background-size: cover !important; }}
.glass-panel {{ background: rgba(255, 255, 255, 0.9) !important; padding: 40px !important; border-radius: 30px !important; }}
"""

def predict(w, e):
    vec = pd.DataFrame(0, index=[0], columns=existing_columns)
    for f in w: vec[f] = 5
    for f in e: vec[f] = 0
    p = model.predict(vec)[0]
    return gr.update(visible=False), gr.update(visible=True), f"{p[0]:.4f}", f"{p[1]:.3f} bar", "3%"

with gr.Blocks(css=custom_css) as demo:
    with gr.Column(elem_classes="glass-panel") as c1:
        gr.Markdown("# 🍃 茶香 AI 製程優化系統")
        i1 = gr.CheckboxGroup(aromas, label="✨ 想要增強的香氣")
        i2 = gr.CheckboxGroup(aromas, label="🚫 想要避開的香氣")
        b1 = gr.Button("開始計算建議製程")
    with gr.Column(visible=False, elem_classes="glass-panel") as c2:
        o1 = gr.Textbox(label="建議固液比")
        o2 = gr.Textbox(label="建議壓力")
        o3 = gr.Textbox(label="建議回添比例")
        b2 = gr.Button("重新設定")
        b2.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[c1, c2])
    b1.click(predict, [i1, i2], [c1, c2, o1, o2, o3])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
