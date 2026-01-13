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
    s = f"{val:.1e}"
    base, exp = s.split('e')
    exp_int = int(exp)
    return rf"{base} \times 10^{{{exp_int}}}"

# --- グラフ描画関数 (動画合成・プレビュー共用) ---
def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_max, y_min, y_max):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    if not df_sub.empty:
        ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2)
        ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=50)
    
    ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=16, fontweight='bold')
    ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=14)
    ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=14)
    ax.set_xlim(0, x_max if x_max > 0 else 1)
    yr = max(float(y_max - y_min), 0.001)
    ax.set_ylim(y_min - yr*0.1, y_max + yr*0.1)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

st.set_page_config(page_title="CartGrapher Pro", layout="wide")
st.title("🚀 CartGrapher Studio: 総合物理解析システム")

# サイドバー
st.sidebar.header("実験パラメータ")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov"])

if "df" not in st.session_state: st.session_state.df = None

if uploaded_file is not None:
    if "file_name" not in st.session_state or st.session_state.file_nameimport streamlit as st
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

# --- 科学表記フォーマッタ (LaTeX用) ---
def format_sci_latex(val):
    """有効数字2桁の科学表記をLaTeX形式で返す"""
    s = f"{val:.1e}"
    base, exp = s.split('e')
    exp_int = int(exp)
    return rf"{base} \times 10^{{{exp_int}}}"

# --- グラフ描画関数 (標準) ---
def create_standard_graph(df, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    if not df.empty:
        ax.plot(df[x_col], df[y_col], color=color, linewidth=2)
        ax.scatter(df[x_col].iloc[-1], df[y_col].iloc[-1], color=color, s=50)
    
    ax.set_title(f"${y_label}$ - ${x_label}$", fontsize=16, fontweight='bold')
    ax.set_xlabel(f"${x_label}$ [{x_unit}]", fontsize=14)
    ax.set_ylabel(f"${y_label}$ [{y_unit}]", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor='white')
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    return cv2.resize(img, (size, size))

# --- グラフ描画関数 (F-x 積分表示用) ---
def create_work_graph(df, x_start, x_end, size):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    ax.plot(df["x"], df["F"], color="purple", linewidth=2, label="Force")
    
    # 積分範囲の塗りつぶし
    df_work = df[(df["x"] >= x_start) & (df["x"] <= x_end)].sort_values("x")
    if len(df_work) > 1:
        ax.fill_between(df_work["x"], df_work["F"], color="purple", alpha=0.3, label="Work (Area)")
    
    ax.set_title(r"$F - x$ Graph", fontsize=16, fontweight='bold')
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

# --- アプリケーションUI ---
st.set_page_config(page_title="CartGrapher Pro", layout="wide")
st.title("🚀 CartGrapher Studio: 物理実験解析システム")

# サイドバー
st.sidebar.header("実験パラメータ設定")
radius_cm = st.sidebar.slider("車輪の半径 (cm)", 0.5, 5.0, 1.6, 0.1)
mass_input = st.sidebar.number_input("台車の質量 m (kg)", value=0.100, min_value=0.001, format="%.3f")
mask_size = st.sidebar.slider("解析エリア半径 (px)", 50, 400, 200, 10)

LOWER_GREEN = (np.array([35, 50, 50]), np.array([85, 255, 255]))
LOWER_PINK = (np.array([140, 40, 40]), np.array([180, 255, 255]))

uploaded_file = st.file_uploader("実験動画をアップロード (MP4/MOV)", type=["mp4", "mov"])

# セッション状態管理
if "df" not in st.session_state: st.session_state.df = None
if "file_id" not in st.session_state: st.session_state.file_id = None

if uploaded_file is not None:
    # ファイルが変更されたら解析をリセット
    if st.session_state.file_id != uploaded_file.name:
        st.session_state.df = None
        st.session_state.file_id = uploaded_file.name

    # --- Step 1: 動画解析 (初回のみ) ---
    if st.session_state.df is None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        
        data_log = []
        total_angle, prev_angle = 0.0, None
        gx, gy = np.nan, np.nan
        
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 中心(緑)
            mask_g = cv2.inRange(hsv, LOWER_GREEN[0], LOWER_GREEN[1])
            con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if con_g:
                c = max(con_g, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0: gx, gy = M["m10"]/M["m00"], M["m01"]/M["m00"]

            # 外周(ピンク)
            bx, by = np.nan, np.nan
            if pd.notna(gx):
                m_circle = np.zeros((h_orig, w_orig), dtype=np.uint8)
                cv2.circle(m_circle, (int(gx), int(gy)), mask_size, 255, -1)
                mask_p = cv2.inRange(cv2.bitwise_and(hsv, hsv, mask=m_circle), LOWER_PINK[0], LOWER_PINK[1])
                con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if con_p:
                    cp = max(con_p, key=cv2.contourArea)
                    Mp = cv2.moments(cp)
                    if Mp["m00"] != 0: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]

            if pd.notna(gx) and pd.notna(bx):
                curr_a = np.arctan2(by - gy, bx - gx)
                if prev_angle is not None:
                    diff = curr_a - prev_angle
                    if diff > np.pi: diff -= 2 * np.pi
                    if diff < -np.pi: diff += 2 * np.pi
                    total_angle += diff 
                prev_angle = curr_a

            data_log.append({"t": f_idx/fps, "x": total_angle * (radius_cm/100)})
            if f_idx % 10 == 0:
                progress_bar.progress(min(f_idx / total_frames, 1.0))
        
        cap.release()
        os.remove(tfile.name)

        # 物理量計算
        df = pd.DataFrame(data_log).interpolate().ffill().bfill()
        df["x"] = savgol_filter(df["x"], 15, 2)
        df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 31, 2)
        df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 31, 2)
        df["F"] = mass_input * df["a"]
        
        st.session_state.df = df
        progress_text.success("✅ 動画の解析が完了しました。")

    # --- Step 2: プレビュー表示 ---
    df = st.session_state.df
    st.subheader("📊 運動のプレビュー")
    ps = 400
    c1, c2, c3 = st.columns(3)
    with c1: st.image(create_standard_graph(df, "t", "x", "t", "x", "s", "m", "blue", ps), channels="BGR")
    with c2: st.image(create_standard_graph(df, "t", "v", "t", "v", "s", "m/s", "red", ps), channels="BGR")
    with c3: st.image(create_standard_graph(df, "t", "a", "t", "a", "s", "m/s^2", "green", ps), channels="BGR")

    # --- Step 3: 仕事 W の計算セクション ---
    st.divider()
    st.subheader("🔬 エネルギー解析: 仕事 $W$ と $\Delta K$")
    
    x_min_val, x_max_val = float(df["x"].min()), float(df["x"].max())
    
    col_ctrl, col_res = st.columns([1, 1])
    
    with col_ctrl:
        st.write("**積分範囲を指定 (変位 x):**")
        x_start = st.number_input("開始点 $x_1$ [m]", value=x_min_val, min_value=x_min_val, max_value=x_max_val, step=0.01)
        x_end = st.number_input("終了点 $x_2$ [m]", value=x_max_val, min_value=x_min_val, max_value=x_max_val, step=0.01)
        
        st.image(create_work_graph(df, x_start, x_end, 500), channels="BGR")

    with col_res:
        df_w = df[(df["x"] >= x_start) & (df["x"] <= x_end)].sort_values("x")
        if len(df_w) > 1:
            # 仕事 W
            work_val = np.trapz(df_w["F"].values, df_w["x"].values)
            # 運動エネルギー変化 ΔK
            v1, v2 = df_w["v"].iloc[0], df_w["v"].iloc[-1]
            dk_val = 0.5 * mass_input * (v2**2 - v1**2)

            st.write("### 計算結果")
            # 科学表記 (2桁)
            st.metric(label="仕事 $W$", value=f"{work_val:.1e} J".replace("e", " × 10^"))
            
            st.write("---")
            st.write("**運動エネルギーの変化:**")
            st.latex(rf"\Delta K = {format_sci_latex(dk_val)} \, \text{{J}}")
            
            # 教育的な比較
            diff = abs(work_val - dk_val)
            st.info(f"仕事とエネルギー変化の差: {diff:.1e} J")
        else:
            st.warning("有効な範囲を指定してください（開始点 < 終了点）。")

    # --- 保存 ---
    st.divider()
    st.download_button("📊 CSVデータを保存", df.to_csv(index=False).encode('utf_8_sig'), "kinema_cart_data.csv")
