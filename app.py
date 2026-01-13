# --- (中略：解析ロジックなどはそのまま) ---

    # --- 表示・計算セクション ---
    df = st.session_state.df_final
    
    st.divider()
    st.subheader("🔬 仕事 $W$ の算出 (F-x グラフの積分)")
    
    col_input, col_graph = st.columns([1, 2])
    
    with col_input:
        x_min_data = float(df["x"].min())
        x_max_data = float(df["x"].max())
        
        st.write("積分範囲の指定:")
        x_start = st.number_input("開始位置 $x_1$ [m]", value=x_min_data, min_value=x_min_data, max_value=x_max_data, step=0.01)
        x_end = st.number_input("終了位置 $x_2$ [m]", value=x_max_data, min_value=x_min_data, max_value=x_max_data, step=0.01)
        
        # 積分計算
        df_w = df[(df["x"] >= x_start) & (df["x"] <= x_end)].sort_values("x")
        
        if len(df_w) > 1:
            # 仕事 W の算出
            work_joule = np.trapz(df_w["F"].values, df_w["x"].values)
            
            # 運動エネルギー変化 ΔK の算出
            v1, v2 = df_w["v"].iloc[0], df_w["v"].iloc[-1]
            delta_k = 0.5 * mass_input * (v2**2 - v1**2)

            # --- 科学表記（有効数字2桁）での表示 ---
            # :.1e は「小数第1位まで表示＋指数部分」なので合計2桁になります
            st.metric(label="仕事 $W$", value=f"{work_joule:.1e} J".replace("e", " × 10^"))
            
            st.write("---")
            st.write(f"**運動エネルギー変化 $\Delta K$**")
            st.latex(rf"\Delta K = {delta_k:.1e} \, \text{{J}}".replace("e", r" \times 10^{") + "}")
            
            # 誤差の確認（教育的なおまけ）
            error = abs(work_joule - delta_k)
            st.caption(f"差分: {error:.1e} J")
        else:
            st.warning("有効な範囲を選択してください。")

    with col_graph:
        fx_img = create_fx_graph_with_work(df, x_start, x_end, 500)
        st.image(fx_img, channels="BGR", caption="紫色のエリアが積分された『仕事』の量です")

# --- (以下、CSV保存などはそのまま) ---
