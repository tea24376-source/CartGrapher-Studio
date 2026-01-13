import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import matplotlib.pyplot as plt
import io

# --- 基本設定 ---
plt.switch_backend('Agg')
plt.rcParams['mathtext.fontset'] = 'cm'
RADIUS_M = 0.016  # 1.6cm固定

def format_sci_latex(val):
    try:
        if abs(val) < 1e-4 and val != 0: # 極端に小さい値の処理
            return "0"
        s = f"{val:.1e}"
        base, exp = s.split('e')
        exp_int = int(exp)
        return rf"{base} \times 10^{{{exp_int}}}"
    except:
        return "0"

# --- グラフ描画関数 ---
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_max, y_min, y_max, shade_range=None):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2, alpha=0.8)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=60, edgecolors='white', zorder=5)
            
            if shade_range is not None and y_col == 'F':
                x1, x2 = shade_range
                # 積分範囲のデータを抽出して塗りつぶし
                mask = (df_sub[x_col] >= x1) & (df_sub[x_col] <= x2)
                ax.fill_between(df_sub[x_col], df_sub[y_col], where=mask, color=color, alpha=0.3)
        
        ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=14, fontweight='bold')
        ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=11)
        ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=11)
        ax.set_xlim(0, max(float(x_max), 0.1))
        yr = max(float(y_max - y_min), 0.01)
        ax.set_ylim(y_min - yr*0.1, y_max + yr*0.1)
        ax.grid(True, linestyle='--', alpha=0.5)
    except: pass
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

st.set_page_config(page_title="Kinema-Cart Studio", layout="wide")
st.title("🚀 CartGrapher Studio")

# サイドバー
st.sidebar.header("解析設定")
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

if uploaded_file:
    if "df" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("映像を解析中..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            data_log = []
            total_angle, prev_angle = 0.0, None
            gx, gy = np.nan, np.nan
            L_G, L_P = (np.array([35,50,50]), np.array([85,255,255])), (np.array([140,40,40]), np.array([180,255,255]))

            for f_idx in range(total):
                ret, frame = cap.read()
                if not ret: break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_g = cv2.inRange(hsv, L_G[0], L_G[1])
                con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if con_g:
                    c = max(con_g, key=cv2.contourArea); M = cv2.moments(c)
                    if M["m00"]!=0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                bx, by = np.nan, np.nan
                if not np.isnan(gx):
                    mc = np.zeros((h, w), dtype=np.uint8); cv2.circle(mc, (int(gx), int(gy)), mask_size, 255, -1)
                    mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=mc), L_P[0], L_P[1])
                    con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if con_p:
                        cp = max(con_p, key=cv2.contourArea); Mp = cv2.moments(cp)
                        if Mp["m00"]!=0: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]
                if not np.isnan(gx) and not np.isnan(bx):
                    curr_a = np.arctan2(by - gy, bx - gx)
                    if prev_angle is not None:
                        diff = curr_a - prev_angle
                        if diff > np.pi: diff -= 2*np.pi
                        elif diff < -np.pi: diff += 2*np.pi
                        total_angle += diff
                    prev_angle = curr_a
                data_log.append({"t": f_idx/fps, "x": total_angle*RADIUS_M, "gx": gx, "gy": gy, "bx": bx, "by": by})
            
            cap.release()
            df = pd.DataFrame(data_log).interpolate().ffill().bfill()
            if len(df) > 31:
                df["x"] = savgol_filter(df["x"], 15, 2)
                df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 31, 2)
                df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 31, 2)
                df["F"] = mass_input * df["a"]
            st.session_state.df = df
            st.session_state.video_meta = {"fps": fps, "w": w, "h": h, "path": tfile.name}
            st.session_state.file_id = uploaded_file.name

    df = st.session_state.df
    st.divider()

    # --- インタラクティブ表示 (2x2) ---
    st.subheader("🖱️ タイムライン・プレビュー")
    time_idx = st.slider("時間をスキャン", 0, len(df)-1, 0)
    curr_row = df.iloc[time_idx]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("積分範囲 (F-x)")
    x1_in = st.sidebar.number_input("開始 x1 [m]", value=float(df["x"].min()))
    x2_in = st.sidebar.number_input("終了 x2 [m]", value=float(df["x"].max()))

    t_max, x_max = df["t"].max(), df["x"].max()
    v_min, v_max = df["v"].min(), df["v"].max()
    a_min, a_max = df["a"].min(), df["a"].max()
    F_min, F_max = df["F"].min(), df["F"].max()
    ps = 450 # グラフサイズ

    # 2x2 レイアウト
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "x", "t", "x", "s", "m", 'blue', ps, t_max, 0, x_max), channels="BGR")
        st.latex(rf"x = {curr_row['x']:.3f} \, \text{{m}}")
    with r1c2:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "v", "t", "v", "s", "m/s", 'red', ps, t_max, v_min, v_max), channels="BGR")
        st.latex(rf"v = {curr_row['v']:.3f} \, \text{{m/s}}")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "a", "t", "a", "s", "m/s^2", 'green', ps, t_max, a_min, a_max), channels="BGR")
        st.latex(rf"a = {curr_row['a']:.3f} \, \text{{m/s}}^2")
    with r2c2:
        st.image(create_graph_image(df.iloc[:time_idx+1], "x", "F", "x", "F", "m", "N", 'purple', ps, x_max, F_min, F_max, shade_range=(x1_in, x2_in)), channels="BGR")
        st.latex(rf"F = {curr_row['F']:.3f} \, \text{{N}}")

    # 仕事とエネルギー
    st.divider()
    df_w = df[(df["x"] >= x1_in) & (df["x"] <= x2_in)].sort_values("x")
    if len(df_w) > 1:
        w_val = np.trapz(df_w["F"], df_w["x"])
        dk_val = 0.5 * mass_input * (df_w["v"].iloc[-1]**2 - df_w["v"].iloc[0]**2)
        cola, colb = st.columns(2)
        cola.latex(rf"W = {format_sci_latex(w_val)} \, \text{{J}}")
        colb.latex(rf"\Delta K = {format_sci_latex(dk_val)} \, \text{{J}}")

    # --- 動画合成 ---
    if st.button("🎥 解析動画を生成"):
        meta = st.session_state.video_meta
        final_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        v_size = meta["w"] // 4
        header_h = v_size + 120
        # cv2.FONT_HERSHEY_SIMPLEX に変更（AttributeError回避）
        font_style = cv2.FONT_HERSHEY_SIMPLEX
        
        out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), meta["fps"], (meta["w"], meta["h"] + header_h))
        cap = cv2.VideoCapture(meta["path"])
        p_bar = st.progress(0.0)

        for i in range(len(df)):
            ret, frame = cap.read()
            if not ret: break
            canvas = np.zeros((meta["h"] + header_h, meta["w"], 3), dtype=np.uint8)
            curr = df.iloc[i]; df_s = df.iloc[:i+1]
            graphs = [('t','x','blue','s','m', 0, x_max), ('t','v','red','s','m/s', v_min, v_max),
                      ('t','a','green','s','m/s2', a_min, a_max), ('x','F','purple','m','N', F_min, F_max)]
            
            for idx, (xc, yc, col, xu, yu, ymi, yma) in enumerate(graphs):
                sr = (x1_in, x2_in) if yc == 'F' else None
                g_img = create_graph_image(df_s, xc, yc, xc, yc, xu, yu, col, v_size, (x_max if xc=='x' else t_max), ymi, yma, shade_range=sr)
                canvas[0:v_size, idx*v_size:(idx+1)*v_size] = g_img
                
                # 瞬間値の描画 (中央下)
                val_text = f"{curr[yc]:>+7.3f} {yu}"
                (tw, th), _ = cv2.getTextSize(val_text, font_style, 0.7, 2)
                tx = idx*v_size + (v_size - tw)//2
                cv2.putText(canvas, val_text, (tx, v_size + 60), font_style, 0.7, (255,255,255), 2)

            # 解析エリア（円）の表示
            if not np.isnan(curr['gx']):
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), mask_size, (255,255,0), 2)
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 5, (0,255,0), -1)
                if not np.isnan(curr['bx']):
                    cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 5, (255,0,255), -1)
            
            canvas[header_h:, :] = frame
            out.write(canvas)
            if i % 20 == 0: p_bar.progress(min(i/len(df), 1.0))
            
        cap.release(); out.release()
        with open(final_path, "rb") as f:
            st.download_button("🎥 解析動画をダウンロード", f, "physics_analysis.mp4")
