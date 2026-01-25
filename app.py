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
VERSION = "3.2.0_MatteColor_Debug" 
MAX_ANALYSIS_WIDTH = 1280

def create_graph_image(df_sub, x_col, y_col, x_label, y_label, x_unit, y_unit, color, size, x_max, y_min, y_max, shade_range=None, markers=None):
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    try:
        if not df_sub.empty:
            ax.plot(df_sub[x_col], df_sub[y_col], color=color, linewidth=2, alpha=0.8)
            ax.scatter(df_sub[x_col].iloc[-1], df_sub[y_col].iloc[-1], color=color, s=60, edgecolors='white', zorder=5)
            
            if markers is not None:
                for t_val in markers:
                    m_row = df_sub.iloc[(df_sub['t']-t_val).abs().argsort()[:1]]
                    if not m_row.empty:
                        ax.scatter(m_row[x_col], m_row[y_col], color='orange', s=50, marker='o', edgecolors='black', zorder=10)

            if shade_range is not None and y_col == 'F':
                t_s, t_e = shade_range
                mask = (df_sub['t'] >= t_s) & (df_sub['t'] <= t_e)
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

def format_sci_latex(val):
    try:
        if abs(val) < 1e-6 and val != 0: return "0"
        s = f"{val:.2e}"
        base, exp = s.split('e')
        exp_int = int(exp)
        if exp_int == 0: return f"{float(base):.2f}"
        return rf"{base} \times 10^{{{exp_int}}}"
    except: return "0"

st.set_page_config(page_title=f"CartGrapher Studio v{VERSION}", layout="wide")
st.title(f"🚀 CartGrapher Studio ver {VERSION}")

# --- サイドバー設定 ---
st.sidebar.header("解析設定")
mass_input = st.sidebar.number_input("台車の質量 $m$ [kg]", value=0.100, min_value=0.001, format="%.3f", step=0.001)

st.sidebar.markdown("---")
st.sidebar.subheader("色認識の調整 (HSV)")
st.sidebar.info("認識がうまくいかない場合、ここの数値を変更して「マスク確認」タブで確認してください。")

# 色調整スライダー
# ピンク (薄いピンク対応: 彩度Sを低く設定)
st.sidebar.markdown("**ピンクマーカー (外側)**")
p_h_min = st.sidebar.slider("Pink Hue Min", 0, 180, 140)
p_h_max = st.sidebar.slider("Pink Hue Max", 0, 180, 180)
p_s_min = st.sidebar.slider("Pink Sat Min (彩度)", 0, 255, 30) # パステル対応でデフォルト下げ
p_v_min = st.sidebar.slider("Pink Val Min (明度)", 0, 255, 100)

# 緑 (暗い緑対応)
st.sidebar.markdown("**緑マーカー (中心)**")
g_h_min = st.sidebar.slider("Green Hue Min", 0, 180, 35)
g_h_max = st.sidebar.slider("Green Hue Max", 0, 180, 95)
g_s_min = st.sidebar.slider("Green Sat Min", 0, 255, 40)
g_v_min = st.sidebar.slider("Green Val Min", 0, 255, 50)

# 白ホイール
st.sidebar.markdown("**白ホイール (全体)**")
w_s_max = st.sidebar.slider("White Sat Max (彩度上限)", 0, 255, 60) # 白は彩度が低い
w_v_min = st.sidebar.slider("White Val Min (明度下限)", 0, 255, 80) # 暗い白も拾う

uploaded_file = st.file_uploader("動画をアップロード (MP4/MOV)", type=["mp4", "mov"])

if uploaded_file:
    tfile_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile_temp.write(uploaded_file.read())
    tfile_temp.close()

    # タブを作成
    tab1, tab2 = st.tabs(["📊 解析結果", "🛠 マスク確認 (デバッグ)"])

    # --- 動画の読み込み準備 ---
    cap = cv2.VideoCapture(tfile_temp.name)
    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fps = raw_fps * 4 
    
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scale_factor = 1.0
    if raw_w > MAX_ANALYSIS_WIDTH:
        scale_factor = MAX_ANALYSIS_WIDTH / raw_w
    w = int(raw_w * scale_factor)
    h = int(raw_h * scale_factor)

    # 色閾値の配列化
    L_P = (np.array([p_h_min, p_s_min, p_v_min]), np.array([p_h_max, 255, 255]))
    L_G = (np.array([g_h_min, g_s_min, g_v_min]), np.array([g_h_max, 255, 255]))
    L_W = (np.array([0, 0, w_v_min]), np.array([180, w_s_max, 255]))

    # --- Tab 2: マスク確認モード ---
    with tab2:
        st.write("動画の最初のフレームで、各色がどのように認識されているか確認できます。白い部分が「認識されている場所」です。")
        if st.button("現在の設定でマスクを確認"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_raw = cap.read()
            if ret:
                frame = cv2.resize(frame_raw, (w, h)) if scale_factor < 1.0 else frame_raw
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # マスク生成
                mask_w = cv2.inRange(hsv, L_W[0], L_W[1])
                mask_p = cv2.inRange(hsv, L_P[0], L_P[1])
                mask_g = cv2.inRange(hsv, L_G[0], L_G[1])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(mask_w, caption="ホイール認識 (白マスク)", clamp=True)
                with col2:
                    st.image(mask_p, caption="ピンクマーカー認識", clamp=True)
                with col3:
                    st.image(mask_g, caption="緑マーカー認識", clamp=True)
                
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="元の画像")
            else:
                st.error("動画を読み込めませんでした。")

    # --- Tab 1: 解析実行 ---
    with tab1:
        if "df" not in st.session_state or st.session_state.get("file_id") != uploaded_file.name or st.button("再解析"):
            with st.spinner("設定に基づき解析中..."):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 先頭に戻す
                data_log = []; total_angle, prev_angle = 0.0, None; last_valid_gx, last_valid_gy = np.nan, np.nan
                
                f_idx = 0
                while True:
                    ret, frame_raw = cap.read()
                    if not ret: break
                    
                    frame = cv2.resize(frame_raw, (w, h)) if scale_factor < 1.0 else frame_raw
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # 1. ホイール領域の検出
                    mask_w = cv2.inRange(hsv, L_W[0], L_W[1])
                    con_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    wheel_roi = np.zeros((h, w), dtype=np.uint8)
                    wheel_detected = False
                    
                    if con_w:
                        # 最大の白領域＝ホイールと仮定
                        c_wheel = max(con_w, key=cv2.contourArea)
                        if cv2.contourArea(c_wheel) > 300: # ノイズ除去閾値
                            cv2.drawContours(wheel_roi, [c_wheel], -1, 255, -1)
                            wheel_detected = True

                    gx, gy, bx, by = np.nan, np.nan, np.nan, np.nan

                    if wheel_detected:
                        # 2. 緑の検出 (ホイール領域内のみ)
                        mask_g = cv2.bitwise_and(cv2.inRange(hsv, L_G[0], L_G[1]), wheel_roi)
                        con_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        new_gx, new_gy = np.nan, np.nan
                        if con_g:
                            c = max(con_g, key=cv2.contourArea)
                            M = cv2.moments(c)
                            if M["m00"] > 10: new_gx, new_gy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                        
                        # 飛び値対策
                        if not np.isnan(last_valid_gx) and not np.isnan(new_gx):
                            if np.sqrt((new_gx - last_valid_gx)**2 + (new_gy - last_valid_gy)**2) > 100: 
                                new_gx, new_gy = last_valid_gx, last_valid_gy
                        
                        gx = new_gx if not np.isnan(new_gx) else last_valid_gx
                        gy = new_gy if not np.isnan(new_gy) else last_valid_gy
                        last_valid_gx, last_valid_gy = gx, gy

                        # 3. ピンクの検出 (ホイール領域内のみ)
                        mask_p = cv2.bitwise_and(cv2.inRange(hsv, L_P[0], L_P[1]), wheel_roi)
                        con_p, _ = cv2.findContours(mask_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if con_p:
                            cp = max(con_p, key=cv2.contourArea)
                            Mp = cv2.moments(cp)
                            if Mp["m00"] > 10: bx, by = Mp["m10"]/Mp["m00"], Mp["m01"]/Mp["m00"]

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
                
                if not data_log:
                     st.error("データが取得できませんでした。")
                     st.stop()

                df = pd.DataFrame(data_log).interpolate().ffill().bfill()
                
                # 平滑化処理
                if len(df) > 31:
                    df["x"] = savgol_filter(df["x"], 15, 2)
                    df["v"] = savgol_filter(df["x"].diff().fillna(0)*fps, 31, 2)
                    df["a"] = savgol_filter(df["v"].diff().fillna(0)*fps, 31, 2)
                    df["F"] = mass_input * df["a"]
                
                st.session_state.df = df 
                st.session_state.video_meta = {"fps": fps, "raw_fps": raw_fps, "w": w, "h": h, "path": tfile_temp.name, "scale": scale_factor}
                st.session_state.file_id = uploaded_file.name

        # --- 以下、グラフ描画と動画生成UI ---
        if "df" in st.session_state:
            df = st.session_state.df
            
            # スライダー範囲設定
            t_max_limit = float(df["t"].max())
            st.markdown("### 範囲選択")
            c_t1, c_t2 = st.columns(2)
            t1 = c_t1.number_input(r"開始時刻 $t_1$ [s]", 0.0, t_max_limit, 0.0, 0.01)
            t2 = c_t2.number_input(r"終了時刻 $t_2$ [s]", 0.0, t_max_limit, t_max_limit, 0.01)
            
            # データ表示
            time_list = [round(t, 4) for t in df["t"].tolist()]
            selected_t = st.select_slider("時刻をスキャン [s]", options=time_list, value=time_list[0])
            time_idx = time_list.index(selected_t); curr_row = df.iloc[time_idx]
            
            # グラフの最大最小値
            t_m, x_m = float(df["t"].max()), float(df["x"].max())
            v_mi, v_ma = float(df["v"].min()), float(df["v"].max())
            a_mi, a_ma = float(df["a"].min()), float(df["a"].max())
            f_mi, f_ma = float(df["F"].min()), float(df["F"].max())

            marker_times = [t1, t2]

            # 4つのグラフを表示
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.image(create_graph_image(df.iloc[:time_idx+1], "t", "x", "t", "x", "s", "m", 'blue', 450, t_m, 0.0, x_m, markers=marker_times), channels="BGR")
                st.latex(rf"x = {curr_row['x']:.3f} \,\, \mathrm{{m}}")
            with r1c2:
                st.image(create_graph_image(df.iloc[:time_idx+1], "t", "v", "t", "v", "s", "m/s", 'red', 450, t_m, v_mi, v_ma, markers=marker_times), channels="BGR")
                st.latex(rf"v = {curr_row['v']:.3f} \,\, \mathrm{{m/s}}")

            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.image(create_graph_image(df.iloc[:time_idx+1], "t", "a", "t", "a", "s", "m/s²", 'green', 450, t_m, a_mi, a_ma, markers=marker_times), channels="BGR")
                st.latex(rf"a = {curr_row['a']:.3f} \,\, \mathrm{{m/s^2}}")
            with r2c2:
                st.image(create_graph_image(df.iloc[:time_idx+1], "x", "F", "x", "F", "m", "N", 'purple', 450, x_m, f_mi, f_ma, shade_range=(t1, t2), markers=marker_times), channels="BGR")
                st.latex(rf"F = {curr_row['F']:.3f} \,\, \mathrm{{N}}")

            st.divider()
            df_w = df[(df["t"] >= t1) & (df["t"] <= t2)]
            if len(df_w) > 1:
                w_val = np.trapezoid(df_w["F"], df_w["x"]) if hasattr(np, 'trapezoid') else np.trapz(df_w["F"], df_w["x"])
                st.latex(rf"W = {format_sci_latex(w_val)} \,\, \mathrm{{J}}")

            if st.button(f"🎥 解析動画を生成して保存"):
                meta = st.session_state.video_meta
                final_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                v_size, header_h = meta["w"] // 4, (meta["w"] // 4) + 100
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                graph_configs = [
                    {"xc": "t", "yc": "x", "xl": "t", "yl": "x", "xu": "s", "yu": "m", "col": "blue", "ymn": 0.0, "ymx": x_m, "xm": t_m},
                    {"xc": "t", "yc": "v", "xl": "t", "yl": "v", "xu": "s", "yu": "m/s", "col": "red", "ymn": v_mi, "ymx": v_ma, "xm": t_m},
                    {"xc": "t", "yc": "a", "xl": "t", "yl": "a", "xu": "s", "yu": "m/s²", "yu_cv": "m/s^2", "col": "green", "ymn": a_mi, "ymx": a_ma, "xm": t_m},
                    {"xc": "x", "yc": "F", "xl": "x", "yl": "F", "xu": "m", "yu": "N", "col": "purple", "ymn": f_mi, "ymx": f_ma, "xm": x_m}
                ]

                out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*'mp4v'), meta["raw_fps"], (meta["w"], meta["h"] + header_h))
                cap_v = cv2.VideoCapture(meta["path"])
                p_bar = st.progress(0.0)
                status_text = st.empty()
                
                for i in range(len(df)):
                    ret, frame_raw = cap_v.read()
                    if not ret: break
                    frame = cv2.resize(frame_raw, (meta["w"], meta["h"])) if meta.get("scale", 1.0) < 1.0 else frame_raw
                    canvas = np.zeros((meta["h"] + header_h, meta["w"], 3), dtype=np.uint8)
                    curr, df_s = df.iloc[i], df.iloc[:i+1]
                    
                    for idx, g in enumerate(graph_configs):
                        canvas[0:v_size, idx*v_size:(idx+1)*v_size] = create_graph_image(df_s, g["xc"], g["yc"], g["xl"], g["yl"], g["xu"], g["yu"], g["col"], v_size, g["xm"], g["ymn"], g["ymx"])
                        val_text = f"{g['yl']} = {curr[g['yc']]:>+7.3f} {g.get('yu_cv', g['yu'])}"
                        tw, _ = cv2.getTextSize(val_text, font, 0.5, 1)[0]
                        cv2.putText(canvas, val_text, (idx*v_size + (v_size-tw)//2, v_size + 50), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    
                    cv2.putText(frame, f"t = {curr['t']:.2f} s", (20, 40), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
                    if not np.isnan(curr['gx']):
                        cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 8, (0,255,0), -1)
                        cv2.circle(frame, (int(curr['gx']), int(curr['gy'])), 8, (255,255,255), 1)
                    if not np.isnan(curr['bx']):
                        cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 8, (255,0,255), -1)
                        cv2.circle(frame, (int(curr['bx']), int(curr['by'])), 8, (255,255,255), 1)
                    
                    canvas[header_h:, :] = frame
                    out.write(canvas)
                    if i % 10 == 0: p_bar.progress(i / len(df)); status_text.text(f"生成中: {i}/{len(df)} フレーム")

                cap_v.release(); out.release(); p_bar.empty(); status_text.success("✅ 動画生成完了")
                with open(final_path, "rb") as f: st.download_button("💾 動画をダウンロード", f, file_name=f"analysis_v{VERSION}.mp4")
