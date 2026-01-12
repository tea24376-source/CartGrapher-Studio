import streamlit as st
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import tempfile
import os
import matplotlib.pyplot as plt
import io

# Matplotlib設定
plt.switch_backend('Agg')
plt.rcParams['mathtext.fontset'] = 'cm'

# --- グラフ描画関数 (積分範囲の塗りつぶし機能付き) ---
def create_fx_graph_with_work(df, x_start, x_end, size):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    
    # 全データ描画
    ax.plot(df["x"], df["F"], color="purple", linewidth=2, label="Force")
    
    # 積分範囲の抽出と塗りつぶし
    df_work = df[(df["x"] >= x_start) & (df["x"] <= x_end)].sort_values("x")
    if len(df_work) > 1:
        ax.fill_between(df_work["x"], df_work["F"], color="purple", alpha=0.3, label="Work (Area)")
    
    ax.set_title(r"$F - x$ Graph (Work Calculation)", fontsize=16, fontweight='bold')
    ax.set_xlabel(r"$x$ [m]", fontsize=14)
    ax.set_ylabel(r"$F$ [N]", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

st.set_page_config(page_title="CartGrapher Pro", layout="wide")
st.title("🚀 CartGrapher Studio: Work & Energy Analysis")

# --- サイドバー設定 ---
st.sidebar.header("実験パラメータ")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("実験動画を選択 (MP4/MOV)", type=["mp4", "mov"])

# --- セッション状態の初期化 ---
if "df_final" not in st.session_state:
    st.session_state.df_final = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

if uploaded_file is not None:
    # 新しいファイルがアップロードされたら解析フラグをリセット
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.df_final = None
        st.session_state.current_file = uploaded_file.name

    # --- Step 1: 解析（未解析の場合のみ実行） ---
    if st.session_state.df_final is None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        status = st.empty()
        progress_bar = st.progress(0.0)
        status.info("映像解析中...（この処理は初回のみ行われます）")
        
        data_log = []
        total_angle = 0.0
        prev_angle = None
        gx, gy = np.nan, np.nan
        
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 中心点トラッキング
            mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
            con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_g:
                c = max(con_g, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]

            # 計測点トラッキング
            bx, by = np.nan, np.nan
            if pd.notna(gx):
                circle_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                cv2.circle(circle_mask, (int(gx), int(gy)), mask_size, 255, -1)
                mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=circle_mask), LOWER_PINK[0], LOWER_PINK[1])
                con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if con_p:
                    cp = max(con_p, key=cv2.contourArea)
                    Mp = cv2.moments(cp)
                    if Mp["m00"] != 0: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]

            if pd.notna(gx) and pd.notna(bx):
                current_angle = np.arctan2(by - gy, bx - gx)
                if prev_angle is not None:
                    diff = current_angle - prev_angle
                    if diff > np.pi: diff -= 2 * np.pi
                    if diff < -np.pi: diff += 2 * np.pi
                    total_angle += diff 
                prev_angle = current_angle

            data_log.append({"t": f_idx/fps, "x": total_angle * (radius_cm/100)})
            if f_idx % 20 == 0: progress_bar.progress(f_idx / total_frames)
        
        cap.release()
        os.remove(tfile.name)

        # 物理量計算
        df = pd.DataFrame(data_log).interpolate().ffill().bfill()
        df["x"] = savgol_filter(df["x"], 15, 2)
        df["v"] = savgol_filter(df["x"].diff().fillna(0) * fps, 31, 2)
        df["a"] = savgol_filter(df["v"].diff().fillna(0) * fps, 31, 2)
        df["F"] = mass_input * df["a"]
        
        st.session_state.df_final = df
        status.success("解析完了！")

    # --- ここから表示・計算セクション (再読み込みされない) ---
    df = st.session_state.df_final
    
    st.divider()
    st.subheader("🔬 仕事 $W$ の算出 (F-x グラフの積分)")
    
    col_input, col_graph = st.columns([1, 2])
    
    with col_input:
        x_min_data = float(df["x"].min())
        x_max_data = float(df["x"].max())
        
        st.write("積分する範囲を指定してください:")
        x_start = st.number_input("開始位置 x1 [m]", value=x_min_data, min_value=x_min_data, max_value=x_max_data, step=0.01)
        x_end = st.number_input("終了位置 x2 [m]", value=x_max_data, min_value=x_min_data, max_value=x_max_data, step=0.01)
        
        # 積分計算 (台形公式)
        df_w = df[(df["x"] >= x_start) & (df["x"] <= x_end)].sort_values("x")
        if len(df_w) > 1:
            work_joule = np.trapz(df_w["F"].values, df_w["x"].values)
            st.metric(label="仕事 W", value=f"{work_joule:.4f} J")
            
            # 運動エネルギーの変化も参考表示
            v1, v2 = df_w["v"].iloc[0], df_w["v"].iloc[-1]
            delta_k = 0.5 * mass_input * (v2**2 - v1**2)
            st.write(f"（参考）運動エネルギー変化 ΔK: `{delta_k:.4f} J`")
        else:
            st.warning("有効な範囲を選択してください。")

    with col_graph:
        # F-xグラフの表示
        fx_img = create_fx_graph_with_work(df, x_start, x_end, 500)
        st.image(fx_img, channels="BGR", caption="F-xグラフの面積が『仕事』を表します")

    st.divider()
    st.download_button("📊 データをCSVで保存", df.to_csv(index=False).encode('utf_8_sig'), "physics_analysis.csv")
