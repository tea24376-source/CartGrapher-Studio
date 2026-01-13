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

# --- 科学表記フォーマッタ ---
def format_sci_latex(val):
    try:
        s = f"{val:.1e}"
        base, exp = s.split('e')
        exp_int = int(exp)
        return rf"{base} \times 10^{{{exp_int}}}"
    except:
        return "0"

# --- グラフ描画関数 ---
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            # color は 'blue', 'red' などの文字列で指定
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
        
        ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=16, fontweight='bold')
        ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=14)
        ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=14)
        
        ax.set_xlim(0, max(float(x_max), 0.1))
        yr = max(float(y_max - y_min), 0.01)
        ax.set_ylim(y_min - yr*0.1, y_max + yr*0.1)
        ax.grid(True, linestyle='--', alpha=0.6)
    except:
        pass
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

st.set_page_config(page_title="CartGrapher Pro", layout="wide")
st.title("🚀 CartGrapher Studio")

# サイドバー
st.sidebar.header("設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

if "df" not in st.session_state: st.session_state.df = None

if uploaded_file is not None:
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        st.session_state.df = None
        st.session_state.file_name = uploaded_file.name

    if st.session_state.df is None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0.0)
        data_log = []
        total_angle, prev_angle = 0.0, None
        gx, gy = np.nan, np.nan
        
        LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
        LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

        for f_idx in range(total):
            ret, frame = cap.read()
            if not ret: break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
            con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_g:
                c = max(con_g, key=cv2.contourArea); M = cv2.moments(c)
                if M["m00"]!=0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
            bx, by = np.nan, np.nan
            if not np.isnan(gx):
                mc = np.zeros((h, w), dtype=np.uint8); cv2.circle(mc, (int(gx), int(gy)), mask_size, 255, -1)
                mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=mc), LOWER_PINK[0], LOWER_PINK[1])
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
            data_log.append({"t": f_idx/fps, "x": total_angle*(radius_cm/100), "gx": gx, "gy": gy, "bx": bx, "by": by})
            if f_idx % 20 == 0: progress_bar.progress(min(f_idx/total, 1.0))
        
        df = pd.DataFrame(data_log).interpolate().ffill().bfill()
        if len(df) > 31:
            df["x"] = savgol_filter(df["x"], 15, 2)
            df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 31, 2)
            df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 31, 2)
            df["F"] = mass_input * df["a"]
        st.session_state.df = df
        st.session_state.video_meta = {"fps": fps, "w": w, "h": h, "path": tfile.name}
        st.success("解析完了")

    df = st.session_state.df
    st.divider()
    st.subheader("🔬 エネルギー解析")
    x1 = st.number_input("開始 x1 [m]", value=float(df["x"].min()))
    x2 = st.number_input("終了 x2 [m]", value=float(df["x"].max()))
    
    df_w = df[(df["x"] >= x1) & (df["x"] <= x2)].sort_values("x")
    if len(df_w) > 1:
        w_val = np.trapz(df_w["F"], df_w["x"])
        dk_val = 0.5 * mass_input * (df_w["v"].iloc[-1]**2 - df_w["v"].iloc[0]**2)
        st.metric("仕事 W", f"{w_val:.1e} J".replace("e", " × 10^"))
        st.latex(rf"\Delta K = {format_sci_latex(dk_val)} \, \text{{J}}")

    if st.button("🎥 解析動画を生成"):
        meta = st.session_state.video_meta
        final_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        v_size = meta["w"] // 4
        header_h = v_size + 100
        out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), meta["fps"], (meta["w"], meta["h"] + header_h))
        
        cap = cv2.VideoCapture(meta["path"])
        p_bar = st.progress(0.0)
        t_max, x_limit = df["t"].max(), df["x"].max()
        v_min, v_max = df["v"].min(), df["v"].max()
        a_min, a_max = df["a"].min(), df["a"].max()
        F_min, F_max = df["F"].min(), df["F"].max()

        for i in range(len(df)):
            ret, frame = cap.read()
            if not ret: break
            canvas = np.zeros((meta["h"] + header_h, meta["w"], 3), dtype=np.uint8)
            curr = df.iloc[i]; df_s = df.iloc[:i+1]
            
            # 色の指定を 'blue', 'red', 'green', 'purple' に修正
            g1 = create_graph_image(df_s, "t", "x", "t", "x", "s", "m", 'blue', v_size, t_max, 0, x_limit)
            g2 = create_graph_image(df_s, "t", "v", "t", "v", "s", "m/s", 'red', v_size, t_max, v_min, v_max)
            g3 = create_graph_image(df_s, "t", "a", "t", "a", "s", "m/s2", 'green', v_size, t_max, a_min, a_max)
            g4 = create_graph_image(df_s, "x", "F", "x", "F", "m", "N", 'purple', v_size, x_limit, F_min, F_max)
            
            canvas[0:v_size, 0:v_size] = g1
            canvas[0:v_size, v_size:v_size*2] = g2
            canvas[0:v_size, v_size*2:v_size*3] = g3
            canvas[0:v_size, v_size*3:v_size*4] = g4
            
            txt = f"t:{curr['t']:.2f}s x:{curr['x']:.3f}m v:{curr['v']:.2f}m/s a:{curr['a']:.2f}m/s2 F:{curr['F']:.3f}N"
            cv2.putText(canvas, txt, (20, v_size + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            if not np.isnan(curr['gx']):
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 6, (0,255,0), -1)
                if not np.isnan(curr['bx']):
                    cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 6, (255,0,255), -1)
            canvas[header_h:, :] = frame
            out.write(canvas)
            if i % 20 == 0: p_bar.progress(min(i/len(df), 1.0))
            
        cap.release(); out.release()
        with open(final_path, "rb") as f:
            st.download_button("🎥 解析済み動画をダウンロード", f, "physics_analysis.mp4")
