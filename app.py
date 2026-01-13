import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import matplotlib.pyplot as plt
import io

# --- 基本設定 ---
plt.switch_backend('Agg')
plt.rcParams['mathtext.fontset'] = 'cm'
RADIUS_M = 0.016
VERSION = "2.1"
MAX_DURATION = 10.0

def format_sci_latex(val):
    try:
        if abs(val) < 1e-6 and val != 0: return "0"
        s = f"{val:.1e}"
        base, exp = s.split('e')
        exp_int = int(exp)
        return rf"{base} \times 10^{{{exp_int}}}"
    except: return "0"

def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_max, y_min, y_max, shade_range=None):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2, alpha=0.8)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=60, edgecolors='white', zorder=5)
            # 時刻ベースの網掛け
            if shade_range is not None and y_col == 'F':
                t_start, t_end = shade_range
                mask = (df_sub['t'] >= t_start) & (df_sub['t'] <= t_end)
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
    return cv2.resize(img, (size, size)) if img is not None else np.zeros((size, size, 3), dtype=np.uint8)

st.set_page_config(page_title=f"CartGrapher Studio v{VERSION}", layout="wide")
st.title(f"🚀 CartGrapher Studio ver {VERSION}")

st.sidebar.header("解析設定")
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f", step=0.001)
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

uploaded_file = st.file_uploader("動画をアップロード (10秒以内)", type=["mp4", "mov"])

if uploaded_file:
    tfile_temp = tempfile.NamedTemporaryFile(delete=False)
    tfile_temp.write(uploaded_file.read())
    cap_check = cv2.VideoCapture(tfile_temp.name)
    fps_check = cap_check.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap_check.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps_check
    cap_check.release()

    if duration > MAX_DURATION:
        st.error(f"❌ 動画時間が長すぎます。")
        st.stop()

    if "df" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("解析中..."):
            cap = cv2.VideoCapture(tfile_temp.name)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            data_log = []
            total_angle, prev_angle = 0.0, None
            last_valid_gx, last_valid_gy = np.nan, np.nan
            L_G, L_P = (np.array([35,50,50]), np.array([85,255,255])), (np.array([140,40,40]), np.array([180,255,255]))
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_g = cv2.inRange(hsv, L_G[0], L_G[1])
                con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                new_gx, new_gy = np.nan, np.nan
                if con_g:
                    c = max(con_g, key=cv2.contourArea); M = cv2.moments(c)
                    if M["m00"] > 100: new_gx, new_gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                if not np.isnan(last_valid_gx) and not np.isnan(new_gx):
                    if np.sqrt((new_gx - last_valid_gx)**2 + (new_gy - last_valid_gy)**2) > 60:
                        new_gx, new_gy = last_valid_gx, last_valid_gy
                if not np.isnan(new_gx): last_valid_gx, last_valid_gy = new_gx, new_gy
                else: new_gx, new_gy = last_valid_gx, last_valid_gy
                gx, gy = new_gx, new_gy
                bx, by = np.nan, np.nan
                if not np.isnan(gx):
                    mc = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mc, (int(gx), int(gy)), mask_size, 255, -1)
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
                f_idx += 1
            cap.release()
            df = pd.DataFrame(data_log).interpolate().ffill().bfill()
            if len(df) > 5:
                df["gx"] = df["gx"].rolling(window=5, center=True).mean().ffill().bfill()
                df["gy"] = df["gy"].rolling(window=5, center=True).mean().ffill().bfill()
            if len(df) > 31:
                df["x"] = savgol_filter(df["x"], 15, 2)
                df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 31, 2)
                df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 31, 2)
                df["F"] = mass_input * df["a"]
            st.session_state.df = df
            st.session_state.video_meta = {"fps": fps, "w": w, "h": h, "path": tfile_temp.name}
            st.session_state.file_id = uploaded_file.name

    df = st.session_state.df
    st.divider()

    # --- 積分範囲の設定 (時刻ベース) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 仕事の計算範囲 (時刻指定)")
    t_max_limit = float(df["t"].max())
    t_start = st.sidebar.number_input("開始時刻 t1 [s]", 0.0, t_max_limit, 0.0, 0.01)
    t_end = st.sidebar.number_input("終了時刻 t2 [s]", 0.0, t_max_limit, t_max_limit, 0.01)

    # 時刻に対応するxを自動取得
    x_start = df.iloc[(df['t']-t_start).abs().argsort()[:1]]['x'].values[0]
    x_end = df.iloc[(df['t']-t_end).abs().argsort()[:1]]['x'].values[0]
    st.sidebar.info(f"自動取得された座標:\n x1: {x_start:.3f} m\n x2: {x_end:.3f} m")

    # プレビュー表示
    time_list = [round(t, 4) for t in df["t"].tolist()]
    selected_t = st.select_slider("時刻を選択 [s]", options=time_list, value=time_list[0])
    time_idx = time_list.index(selected_t)
    curr_row = df.iloc[time_idx]

    t_m, x_m = float(df["t"].max()), float(df["x"].max())
    v_mi, v_ma = float(df["v"].min()), float(df["v"].max())
    a_mi, a_ma = float(df["a"].min()), float(df["a"].max())
    f_mi, f_ma = float(df["F"].min()), float(df["F"].max())

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "x", "t", "x", "s", "m", 'blue', 450, t_m, 0.0, x_m), channels="BGR")
        st.latex(rf"x = {curr_row['x']:.3f} \, \text{{m}}")
    with r1c2:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "v", "t", "v", "s", "m/s", 'red', 450, t_m, v_mi, v_ma), channels="BGR")
        st.latex(rf"v = {curr_row['v']:.3f} \, \text{{m/s}}")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.image(create_graph_image(df.iloc[:time_idx+1], "t", "a", "t", "a", "s", "m/s^2", 'green', 450, t_m, a_mi, a_ma), channels="BGR")
        st.latex(rf"a = {curr_row['a']:.3f} \, \text{{m/s}}^2")
    with r2c2:
        # F-xグラフ（時間指定の網掛け）
        st.image(create_graph_image(df.iloc[:time_idx+1], "x", "F", "x", "F", "m", "N", 'purple', 450, x_m, f_mi, f_ma, shade_range=(t_start, t_end)), channels="BGR")
        st.latex(rf"F = {curr_row['F']:.3f} \, \text{{N}}")

    st.divider()
    
    # --- 時刻順の積分 (Work calculation) ---
    df_w = df[(df["t"] >= t_start) & (df["t"] <= t_end)]
    if len(df_w) > 1:
        # np.trapz(y, x) を時刻順のデータのまま実行（xでソートしない！）
        w_val = np.trapz(df_w["F"], df_w["x"])
        dk_val = 0.5 * mass_input * (df_w["v"].iloc[-1]**2 - df_w["v"].iloc[0]**2)
        cola, colb = st.columns(2)
        cola.latex(rf"W = \int_{{x_1}}^{{x_2}} F dx = {format_sci_latex(w_val)} \, \text{{J}}")
        colb.latex(rf"\Delta K = K_2 - K_1 = {format_sci_latex(dk_val)} \, \text{{J}}")
