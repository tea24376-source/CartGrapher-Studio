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
RADIUS_M = 0.016
VERSION = "1.4"

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
            if shade_range is not None and y_col == 'F':
                x1, x2 = shade_range
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
    return cv2.resize(img, (size, size)) if img is not None else np.zeros((size, size, 3), dtype=np.uint8)

st.set_page_config(page_title=f"CartGrapher Studio v{VERSION}", layout="wide")
st.title(f"🚀 CartGrapher Studio ver {VERSION}")
st.caption("授業用最適化モード：高負荷動画の自動間引き・リサイズ有効")

st.sidebar.header("解析設定")
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

uploaded_file = st.file_uploader("動画をアップロード (1080p/60fps対応)", type=["mp4", "mov"])

if uploaded_file:
    if "df" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name:
        with st.spinner("動画を最適化しながら解析中..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # --- 自動最適化判定 ---
            skip_frames = 1 if orig_fps >= 45 else 0  # 60fpsなら1つ飛ばし(実質30fps)
            calc_fps = orig_fps / (skip_frames + 1)
            
            # 解析用の解像度（負荷軽減のため上限720p程度に制限）
            scale = 1.0
            if orig_w > 1280:
                scale = 1280.0 / orig_w
            
            data_log = []
            total_angle, prev_angle = 0.0, None
            last_valid_gx, last_valid_gy = np.nan, np.nan
            L_G, L_P = (np.array([35,50,50]), np.array([85,255,255])), (np.array([140,40,40]), np.array([180,255,255]))

            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # フレーム間引き
                if skip_frames > 0 and f_idx % (skip_frames + 1) != 0:
                    f_idx += 1
                    continue
                
                # リサイズ
                if scale != 1.0:
                    frame = cv2.resize(frame, (int(orig_w * scale), int(orig_h * scale)))
                
                h, w = frame.shape[:2]
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_g = cv2.inRange(hsv, L_G[0], L_G[1])
                con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                new_gx, new_gy = np.nan, np.nan
                if con_g:
                    c = max(con_g, key=cv2.contourArea); M = cv2.moments(c)
                    if M["m00"] > 100: new_gx, new_gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                
                if not np.isnan(last_valid_gx) and not np.isnan(new_gx):
                    if np.sqrt((new_gx - last_valid_gx)**2 + (new_gy - last_valid_gy)**2) > (60 * scale):
                        new_gx, new_gy = last_valid_gx, last_valid_gy
                
                if not np.isnan(new_gx): last_valid_gx, last_valid_gy = new_gx, new_gy
                else: new_gx, new_gy = last_valid_gx, last_valid_gy
                gx, gy = new_gx, new_gy

                bx, by = np.nan, np.nan
                if not np.isnan(gx):
                    mc = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mc, (int(gx), int(gy)), int(mask_size * scale), 255, -1)
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
                
                # 時間 t は元のインデックスとFPSから正確に算出
                data_log.append({"t": f_idx/orig_fps, "x": total_angle*RADIUS_M, "gx": gx/scale, "gy": gy/scale, "bx": bx/scale, "by": by/scale})
                f_idx += 1

            cap.release()
            df = pd.DataFrame(data_log).interpolate().ffill().bfill()
            if len(df) > 5:
                df["gx"] = df["gx"].rolling(window=5, center=True).mean().ffill().bfill()
                df["gy"] = df["gy"].rolling(window=5, center=True).mean().ffill().bfill()
            if len(df) > 15: # 間引いた分、フィルタ窓を調整
                df["x"] = savgol_filter(df["x"], 11, 2)
                df["v"] = savgol_filter(df["x"].diff().fillna(0)*orig_fps/(skip_frames+1), 15, 2)
                df["a"] = savgol_filter(df["v"].diff().fillna(0)*orig_fps/(skip_frames+1), 15, 2)
                df["F"] = mass_input * df["a"]
            
            st.session_state.df = df
            st.session_state.video_meta = {"fps": orig_fps, "w": orig_w, "h": orig_h, "path": tfile.name, "skip": skip_frames}
            st.session_state.file_id = uploaded_file.name

    df = st.session_state.df
    st.divider()

    # --- プレビュー ---
    st.subheader("🖱️ タイムライン・プレビュー")
    time_idx = st.slider("時間をスキャン", 0, len(df)-1, 0)
    curr_row = df.iloc[time_idx]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("積分範囲 (F-x)")
    x1_in = st.sidebar.number_input("開始 x1 [m]", value=float(df["x"].min()))
    x2_in = st.sidebar.number_input("終了 x2 [m]", value=float(df["x"].max()))

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
        st.image(create_graph_image(df.iloc[:time_idx+1], "x", "F", "x", "F", "m", "N", 'purple', 450, x_m, f_mi, f_ma, shade_range=(x1_in, x2_in)), channels="BGR")
        st.latex(rf"F = {curr_row['F']:.3f} \, \text{{N}}")

    st.divider()
    df_w = df[(df["x"] >= x1_in) & (df["x"] <= x2_in)].sort_values("x")
    if len(df_w) > 1:
        w_val = np.trapz(df_w["F"], df_w["x"])
        dk_val = 0.5 * mass_input * (df_w["v"].iloc[-1]**2 - df_w["v"].iloc[0]**2)
        cola, colb = st.columns(2)
        cola.latex(rf"W = {format_sci_latex(w_val)} \, \text{{J}}")
        colb.latex(rf"\Delta K = {format_sci_latex(dk_val)} \, \text{{J}}")

    # --- 動画合成 ---
    if st.button(f"🎥 ver {VERSION} 動画を生成"):
        meta = st.session_state.video_meta
        final_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        v_size, font = meta["w"] // 4, cv2.FONT_HERSHEY_SIMPLEX
        header_h = v_size + 100
        graph_configs = [
            {"xc": "t", "yc": "x", "col": "blue", "xu": "m", "sym": "x", "ymn": 0.0, "ymx": x_m, "xm": t_m},
            {"xc": "t", "yc": "v", "col": "red", "xu": "m/s", "sym": "v", "ymn": v_mi, "ymx": v_ma, "xm": t_m},
            {"xc": "t", "yc": "a", "col": "green", "xu": "m/s2", "sym": "a", "ymn": a_mi, "ymx": a_ma, "xm": t_m},
            {"xc": "x", "yc": "F", "col": "purple", "xu": "N", "sym": "F", "ymn": f_mi, "ymx": f_ma, "xm": x_m}
        ]
        out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), meta["fps"], (meta["w"], meta["h"] + header_h))
        cap = cv2.VideoCapture(meta["path"])
        p_bar = st.progress(0.0)
        
        # 動画合成時は元の全フレームを読み込むが、データは解析済みのものを使用
        for i in range(len(df)):
            # 解析時に飛ばした分、動画読み込みも進める
            for _ in range(meta["skip"] + 1):
                ret, frame = cap.read()
            if not ret: break
            
            canvas = np.zeros((meta["h"] + header_h, meta["w"], 3), dtype=np.uint8)
            curr = df.iloc[i]; df_s = df.iloc[:i+1]
            for idx, g in enumerate(graph_configs):
                g_img = create_graph_image(df_s, g["xc"], g["yc"], g["xc"], g["yc"], "", "", g["col"], v_size, g["xm"], g["ymn"], g["ymx"], shade_range=None)
                canvas[0:v_size, idx*v_size:(idx+1)*v_size] = g_img
                val_text = f"{g['sym']} = {curr[g['yc']]:>+7.3f} {g['xu']}"
                (tw, th), _ = cv2.getTextSize(val_text, font, 0.55, 2)
                cv2.putText(canvas, val_text, (idx*v_size + (v_size-tw)//2, v_size + 60), font, 0.55, (255,255,255), 2)
            
            if not np.isnan(curr['gx']):
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), mask_size, (255,255,0), 2)
                cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 5, (0,255,0), -1)
                if not np.isnan(curr['bx']): cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 5, (255,0,255), -1)
            
            t_text = f"t = {curr['t']:.2f} s"
            (ttw, tth), _ = cv2.getTextSize(t_text, font, 0.8, 2)
            cv2.putText(frame, t_text, (meta["w"] - ttw - 20, meta["h"] - 30), font, 0.8, (255,255,255), 2, cv2.LINE_AA)
            canvas[header_h:, :] = frame
            out.write(canvas)
            if i % 20 == 0: p_bar.progress(min(i/len(df), 1.0))
        cap.release(); out.release()
        with open(final_path, "rb") as f:
            st.download_button(f"🎥 v{VERSION} 高速解析版を保存", f, f"cart_v{VERSION}_optimized.mp4")
