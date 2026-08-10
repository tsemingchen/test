"""
49th Parallel — Sales Data -> Rates -> Forecast Translator + Dashboard

Landing page is the Dashboard: weekly and monthly actual-vs-forecast reporting.
Upload raw sales records to compute real price/kg and mix % rates, translate
any $ forecast into kg, log actuals weekly to track accuracy, check ops
capacity, and sign off as a consensus plan.

Deploy on Streamlit Community Cloud:
  1. Push this file + requirements.txt to a GitHub repo
  2. Go to https://share.streamlit.io, connect the repo, pick this file
  3. Deploy -> shareable link, no installs needed for either team

Run locally to test first:
  pip install -r requirements.txt
  streamlit run app.py

Storage note: uses a local SQLite file. On Streamlit Community Cloud's free
tier, this can reset when the app is redeployed or wakes from sleep -- use
the CSV backup buttons in the History tab periodically.
"""

import sqlite3
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="49th Parallel — Demand Planning", layout="wide")
DB_PATH = "demand_planning.db"
WEEKS_PER_MONTH = 4.345


# ===================================================================
# DATABASE
# ===================================================================
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS sales_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_batch TEXT, uploaded_at TEXT,
        record_date TEXT, channel TEXT, customer TEXT, product TEXT,
        size_label TEXT, kg REAL, revenue REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS dollar_forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, submitted_by TEXT, cycle_label TEXT,
        channel TEXT, forecast_usd REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS weekly_actuals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, submitted_by TEXT, cycle_label TEXT,
        week_ending TEXT, channel TEXT, product TEXT, actual_kg REAL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS ops_capacity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, submitted_by TEXT, cycle_label TEXT,
        monthly_capacity_kg REAL, notes TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS signoffs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, cycle_label TEXT, role TEXT, name TEXT
    )""")
    conn.commit()
    return conn


conn = get_conn()


def load_sales_records():
    return pd.read_sql("SELECT * FROM sales_records", conn)


def current_cycle_label():
    return date.today().strftime("%Y-%m")


# ===================================================================
# RATE ENGINE — computed live from whatever is in sales_records
# ===================================================================
def compute_price_per_kg(df):
    g = df.groupby(["channel", "product", "size_label"], as_index=False).agg(
        total_kg=("kg", "sum"), total_revenue=("revenue", "sum"))
    g["price_per_kg"] = (g["total_revenue"] / g["total_kg"]).round(2)
    return g[["channel", "product", "size_label", "price_per_kg", "total_kg", "total_revenue"]]


def compute_product_mix(df):
    g = df.groupby(["channel", "product"], as_index=False)["kg"].sum()
    g["channel_total_kg"] = g.groupby("channel")["kg"].transform("sum")
    g["product_mix_pct"] = (g["kg"] / g["channel_total_kg"] * 100).round(1)
    return g[["channel", "product", "product_mix_pct", "kg"]]


def compute_size_mix(df):
    g = df.groupby(["channel", "product", "size_label"], as_index=False)["kg"].sum()
    g["group_total_kg"] = g.groupby(["channel", "product"])["kg"].transform("sum")
    g["size_mix_pct"] = (g["kg"] / g["group_total_kg"] * 100).round(1)
    return g[["channel", "product", "size_label", "size_mix_pct", "kg"]]


def compute_customer_mix(df):
    g = df.groupby(["channel", "product", "customer"], as_index=False)["kg"].sum()
    g["group_total_kg"] = g.groupby(["channel", "product"])["kg"].transform("sum")
    g["customer_mix_pct"] = (g["kg"] / g["group_total_kg"] * 100).round(1)
    return g[["channel", "product", "customer", "customer_mix_pct", "kg"]].sort_values(
        ["channel", "product", "customer_mix_pct"], ascending=[True, True, False])


def translate_forecast(dollar_by_channel, price_df, product_mix_df, size_mix_df):
    """$ forecast by channel -> kg by channel/product/size, using computed rates."""
    rows = []
    for _, fc in dollar_by_channel.iterrows():
        channel, usd = fc["channel"], fc["forecast_usd"]
        pm = product_mix_df[product_mix_df["channel"] == channel]
        for _, p in pm.iterrows():
            product = p["product"]
            product_usd = usd * (p["product_mix_pct"] / 100.0)
            sm = size_mix_df[(size_mix_df["channel"] == channel) & (size_mix_df["product"] == product)]
            for _, s in sm.iterrows():
                size_usd = product_usd * (s["size_mix_pct"] / 100.0)
                price_row = price_df[(price_df["channel"] == channel) & (price_df["product"] == product) &
                                      (price_df["size_label"] == s["size_label"])]
                price = price_row["price_per_kg"].iloc[0] if not price_row.empty else np.nan
                kg = size_usd / price if price and price > 0 else np.nan
                rows.append({"channel": channel, "product": product, "size_label": s["size_label"],
                             "forecast_usd": round(size_usd, 2), "price_per_kg": price,
                             "forecast_kg": round(kg, 1) if pd.notna(kg) else None})
    return pd.DataFrame(rows)


# ===================================================================
# GLOBAL STATE — computed once, available to every tab (incl. Dashboard first)
# ===================================================================
st.title("Sales to operations demand planning")
cycle = st.text_input("Planning cycle label", value=current_cycle_label(),
                       help="e.g. 2026-08. Everyone on the same cycle label shares the same plan.")

sales_df = load_sales_records()
has_data = not sales_df.empty

if has_data:
    price_df = compute_price_per_kg(sales_df)
    product_mix_df = compute_product_mix(sales_df)
    size_mix_df = compute_size_mix(sales_df)
    customer_mix_df = compute_customer_mix(sales_df)
else:
    price_df = product_mix_df = size_mix_df = customer_mix_df = pd.DataFrame()

latest_forecast = pd.read_sql(
    "SELECT * FROM dollar_forecasts WHERE cycle_label = ? AND id IN "
    "(SELECT MAX(id) FROM dollar_forecasts WHERE cycle_label = ? GROUP BY channel)",
    conn, params=(cycle, cycle))

if has_data and not latest_forecast.empty:
    translated = translate_forecast(latest_forecast[["channel", "forecast_usd"]], price_df, product_mix_df, size_mix_df)
    forecast_by_cp = translated.groupby(["channel", "product"], as_index=False)["forecast_kg"].sum()
else:
    translated = pd.DataFrame()
    forecast_by_cp = pd.DataFrame()


# ===================================================================
# TABS — Dashboard first (landing page)
# ===================================================================
tab_dash, tab_data, tab_rates, tab_translate, tab_ops, tab_signoff, tab_history = st.tabs(
    ["Dashboard", "1. Upload sales data", "2. Computed rates", "3. Translate $ forecast",
     "4. Ops capacity check", "5. Sign-off", "6. History"]
)

# --- DASHBOARD (landing page) ---
with tab_dash:
    if not has_data:
        st.info("No sales data uploaded yet. Go to **1. Upload sales data** to get started.")
    elif forecast_by_cp.empty:
        st.warning("Rates are ready, but no $ forecast saved yet for this cycle. "
                   "Go to **3. Translate $ forecast** to save one, then come back here.")
    else:
        view = st.radio("View", ["Weekly report", "Monthly report"], horizontal=True)
        fc_map = {(r["channel"], r["product"]): r["forecast_kg"] / WEEKS_PER_MONTH
                  for _, r in forecast_by_cp.iterrows()}

        if view == "Weekly report":
            with st.expander("Log this week's actuals", expanded=True):
                with st.form("log_actual_form"):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        week_ending = st.date_input("Week ending", value=date.today())
                    with c2:
                        ch = st.selectbox("Channel", sorted(sales_df["channel"].unique().tolist()))
                    with c3:
                        pr = st.selectbox("Product", sorted(sales_df["product"].unique().tolist()))
                    with c4:
                        actual_kg = st.number_input("Actual kg this week", min_value=0.0, step=1.0)
                    submitted_by = st.text_input("Your name")
                    if st.form_submit_button("Log actual", type="primary"):
                        conn.execute("""INSERT INTO weekly_actuals
                            (timestamp, submitted_by, cycle_label, week_ending, channel, product, actual_kg)
                            VALUES (?,?,?,?,?,?,?)""",
                            (datetime.now().isoformat(), submitted_by, cycle, week_ending.isoformat(), ch, pr, actual_kg))
                        conn.commit()
                        st.success("Logged.")
                        st.rerun()

            weekly = pd.read_sql("SELECT * FROM weekly_actuals WHERE cycle_label = ? ORDER BY week_ending",
                                  conn, params=(cycle,))
            if weekly.empty:
                st.info("No actuals logged yet this cycle — log this week's numbers above to start the report.")
            else:
                weekly["forecast_kg"] = weekly.apply(lambda r: fc_map.get((r["channel"], r["product"]), np.nan), axis=1)
                weekly["variance_pct"] = (weekly["actual_kg"] - weekly["forecast_kg"]) / weekly["forecast_kg"]

                combos = weekly[["channel", "product"]].drop_duplicates()
                combos["label"] = combos["channel"] + " — " + combos["product"]
                sel_label = st.selectbox("Chart for", combos["label"].tolist())
                sel_ch, sel_pr = combos.loc[combos["label"] == sel_label, ["channel", "product"]].iloc[0]
                chart_df = weekly[(weekly["channel"] == sel_ch) & (weekly["product"] == sel_pr)] \
                    .sort_values("week_ending").tail(8)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=chart_df["week_ending"], y=chart_df["forecast_kg"],
                                      name="Forecast", marker_color="rgba(139,90,60,0.35)"))
                fig.add_trace(go.Bar(x=chart_df["week_ending"], y=chart_df["actual_kg"],
                                      name="Actual", marker_color="rgb(139,90,60)"))
                fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=10),
                                   xaxis_title="Week ending", yaxis_title="Kg")
                st.plotly_chart(fig, use_container_width=True)

                recent_bias = chart_df["variance_pct"].tail(4).mean()
                n_weeks = min(4, len(chart_df))
                if pd.notna(recent_bias):
                    if abs(recent_bias) > 0.10:
                        direction = "above" if recent_bias > 0 else "below"
                        st.error(f"Trending {abs(recent_bias)*100:.0f}% {direction} forecast "
                                 f"over the last {n_weeks} week(s) — worth flagging.")
                    else:
                        st.success(f"Tracking within {abs(recent_bias)*100:.0f}% of forecast "
                                   f"over the last {n_weeks} week(s).")

                st.dataframe(
                    weekly[["week_ending", "channel", "product", "forecast_kg", "actual_kg", "variance_pct"]]
                    .sort_values("week_ending", ascending=False).round(2),
                    use_container_width=True)

        else:  # Monthly report
            weekly = pd.read_sql("SELECT * FROM weekly_actuals WHERE cycle_label = ?", conn, params=(cycle,))
            if weekly.empty:
                st.info("No actuals logged yet this cycle — switch to Weekly report to log some first.")
            else:
                weekly["forecast_kg"] = weekly.apply(lambda r: fc_map.get((r["channel"], r["product"]), np.nan), axis=1)
                weekly["variance_pct"] = (weekly["actual_kg"] - weekly["forecast_kg"]) / weekly["forecast_kg"]
                weekly["abs_variance_pct"] = weekly["variance_pct"].abs()
                monthly = weekly.groupby(["channel", "product"], as_index=False).agg(
                    MAPE=("abs_variance_pct", "mean"), Bias=("variance_pct", "mean"),
                    weeks_logged=("week_ending", "count"))
                monthly["MAPE_%"] = (monthly["MAPE"] * 100).round(1)
                monthly["Bias_%"] = (monthly["Bias"] * 100).round(1)
                st.dataframe(monthly[["channel", "product", "MAPE_%", "Bias_%", "weeks_logged"]],
                             use_container_width=True)
                st.caption("Positive bias = under-forecasting (actuals running ahead of plan). "
                           "Negative = over-forecasting. MAPE = average error size regardless of direction.")

# --- TAB 1: Upload ---
with tab_data:
    st.subheader("Upload raw sales records")
    st.caption(
        "One row per sale line: channel, customer, product, size, kg sold, revenue. "
        "Export this from Lightspeed/Acumatica and upload as CSV."
    )
    uploaded = st.file_uploader("CSV file", type="csv")

    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(raw.head(), use_container_width=True)

        st.markdown("**Map your columns**")
        cols = list(raw.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            date_col = st.selectbox("Date column", ["(none)"] + cols)
            channel_col = st.selectbox("Channel column", cols)
        with c2:
            customer_col = st.selectbox("Customer column", cols)
            product_col = st.selectbox("Product column", cols)
        with c3:
            size_col = st.selectbox("Size / package column", cols)
            revenue_col = st.selectbox("Revenue ($) column", cols)

        kg_mode = st.radio("How is weight recorded?", ["I have a direct KG column", "I have Units + Weight-per-unit (kg)"])
        if kg_mode == "I have a direct KG column":
            kg_col = st.selectbox("KG column", cols)
        else:
            units_col = st.selectbox("Units column", cols)
            weight_col = st.selectbox("Weight per unit (kg) column", cols)

        batch_name = st.text_input("Name this upload batch", value=f"upload_{datetime.now().strftime('%Y%m%d_%H%M')}")

        if st.button("Process and save", type="primary"):
            std = pd.DataFrame()
            std["record_date"] = raw[date_col].astype(str) if date_col != "(none)" else ""
            std["channel"] = raw[channel_col].astype(str)
            std["customer"] = raw[customer_col].astype(str)
            std["product"] = raw[product_col].astype(str)
            std["size_label"] = raw[size_col].astype(str)
            std["revenue"] = pd.to_numeric(raw[revenue_col], errors="coerce")
            if kg_mode == "I have a direct KG column":
                std["kg"] = pd.to_numeric(raw[kg_col], errors="coerce")
            else:
                std["kg"] = pd.to_numeric(raw[units_col], errors="coerce") * pd.to_numeric(raw[weight_col], errors="coerce")
            std = std.dropna(subset=["kg", "revenue"])
            std["upload_batch"] = batch_name
            std["uploaded_at"] = datetime.now().isoformat()
            std.to_sql("sales_records", conn, if_exists="append", index=False)
            st.success(f"Saved {len(std)} records from batch '{batch_name}'.")
            st.rerun()

    st.divider()
    st.caption(f"Total records in database: {len(sales_df)}")
    if has_data:
        batches = sales_df["upload_batch"].unique().tolist()
        st.write("Batches uploaded so far:", ", ".join(batches))
        if st.button("Clear ALL uploaded records (careful)"):
            conn.execute("DELETE FROM sales_records")
            conn.commit()
            st.warning("All sales records cleared.")
            st.rerun()

# --- TAB 2: Computed rates ---
with tab_rates:
    st.subheader("Rates computed from your uploaded data")
    if not has_data:
        st.warning("No sales data uploaded yet — go to tab 1 first.")
    else:
        st.markdown("**Price per kg** (channel x product x size, weighted average from actual revenue/kg)")
        st.dataframe(price_df, use_container_width=True)
        st.download_button("Download price_per_kg.csv", price_df.to_csv(index=False), "price_per_kg.csv")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Product mix %** (within each channel)")
            st.dataframe(product_mix_df, use_container_width=True)
        with colB:
            st.markdown("**Size mix %** (within each channel-product)")
            st.dataframe(size_mix_df, use_container_width=True)

        st.markdown("**Customer mix %** (within each channel-product, highest share first)")
        st.dataframe(customer_mix_df, use_container_width=True)
        st.download_button("Download customer_mix.csv", customer_mix_df.to_csv(index=False), "customer_mix.csv")

# --- TAB 3: Translate ---
with tab_translate:
    st.subheader("Translate a $ forecast into kg and bags")
    if not has_data:
        st.warning("Upload data in tab 1 first so rates can be computed.")
    else:
        channels = sorted(sales_df["channel"].unique().tolist())
        st.markdown("**Enter your $ forecast by channel for this cycle**")
        with st.form("dollar_forecast_form"):
            submitted_by = st.text_input("Your name")
            forecast_inputs = {}
            for ch in channels:
                forecast_inputs[ch] = st.number_input(f"{ch} — forecast $", min_value=0.0, step=100.0, key=f"fc_{ch}")
            if st.form_submit_button("Save and translate", type="primary"):
                for ch, usd in forecast_inputs.items():
                    conn.execute("""INSERT INTO dollar_forecasts (timestamp, submitted_by, cycle_label, channel, forecast_usd)
                        VALUES (?,?,?,?,?)""", (datetime.now().isoformat(), submitted_by, cycle, ch, usd))
                conn.commit()
                st.success("Forecast saved.")
                st.rerun()

        if not translated.empty:
            st.markdown("**Translated forecast — kg by channel / product / size**")
            st.dataframe(translated, use_container_width=True)
            total_kg = translated["forecast_kg"].sum()
            st.metric("Total forecast kg", f"{total_kg:,.0f} kg" if pd.notna(total_kg) else "n/a")
            st.download_button("Download translated_forecast.csv", translated.to_csv(index=False), "translated_forecast.csv")
        else:
            st.info("No saved $ forecast for this cycle yet.")

# --- TAB 4: Ops capacity check ---
with tab_ops:
    st.subheader("Operations: enter capacity and check against the plan")
    if translated.empty:
        st.warning("Translate a forecast in tab 3 first.")
    else:
        existing_cap = pd.read_sql("SELECT * FROM ops_capacity WHERE cycle_label = ? ORDER BY id DESC LIMIT 1",
                                    conn, params=(cycle,))
        existing_cap = existing_cap.iloc[0] if not existing_cap.empty else None
        with st.form("ops_form"):
            ops_name = st.text_input("Your name", value=existing_cap["submitted_by"] if existing_cap is not None else "")
            monthly_capacity = st.number_input(
                "Monthly production capacity (kg)",
                value=float(existing_cap["monthly_capacity_kg"]) if existing_cap is not None else 4000.0)
            ops_notes = st.text_area("Notes", value=existing_cap["notes"] if existing_cap is not None else "")
            if st.form_submit_button("Save capacity", type="primary"):
                conn.execute("""INSERT INTO ops_capacity (timestamp, submitted_by, cycle_label, monthly_capacity_kg, notes)
                    VALUES (?,?,?,?,?)""", (datetime.now().isoformat(), ops_name, cycle, monthly_capacity, ops_notes))
                conn.commit()
                st.success("Capacity saved.")
                st.rerun()

        cap_row = pd.read_sql("SELECT * FROM ops_capacity WHERE cycle_label = ? ORDER BY id DESC LIMIT 1",
                               conn, params=(cycle,))
        if not cap_row.empty:
            cap = cap_row.iloc[0]["monthly_capacity_kg"]
            planned = translated["forecast_kg"].sum()
            status = "SHORTFALL" if cap < planned else "OK"
            st.metric("Planned kg vs capacity", f"{planned:,.0f} kg planned / {cap:,.0f} kg capacity", status)
            if status == "SHORTFALL":
                st.error(f"Capacity shortfall of {planned - cap:,.0f} kg — flag for sign-off discussion.")
            else:
                st.success("Capacity covers this plan.")

# --- TAB 5: Sign-off ---
with tab_signoff:
    st.subheader("Consensus sign-off")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sales sign-off**")
        sales_signer = st.text_input("Sales rep name", key="sales_signer")
        if st.button("Sign off — Sales"):
            conn.execute("INSERT INTO signoffs (timestamp, cycle_label, role, name) VALUES (?,?,?,?)",
                         (datetime.now().isoformat(), cycle, "Sales", sales_signer))
            conn.commit()
            st.success("Sales sign-off recorded.")
    with col2:
        st.markdown("**Ops sign-off**")
        ops_signer = st.text_input("Ops rep name", key="ops_signer")
        if st.button("Sign off — Ops"):
            conn.execute("INSERT INTO signoffs (timestamp, cycle_label, role, name) VALUES (?,?,?,?)",
                         (datetime.now().isoformat(), cycle, "Ops", ops_signer))
            conn.commit()
            st.success("Ops sign-off recorded.")

    signoffs = pd.read_sql("SELECT * FROM signoffs WHERE cycle_label = ? ORDER BY id DESC", conn, params=(cycle,))
    has_sales = (signoffs["role"] == "Sales").any() if not signoffs.empty else False
    has_ops = (signoffs["role"] == "Ops").any() if not signoffs.empty else False
    if has_sales and has_ops:
        st.success(f"Cycle {cycle} APPROVED by both sides.")
    else:
        st.info(f"Pending: {'Sales OK' if has_sales else 'Sales -'} / {'Ops OK' if has_ops else 'Ops -'}")
    st.dataframe(signoffs, use_container_width=True)

# --- TAB 6: History ---
with tab_history:
    st.subheader("Backup / export everything")
    st.markdown("**Sales records**")
    st.dataframe(sales_df, use_container_width=True)
    st.download_button("Download sales_records.csv", sales_df.to_csv(index=False), "sales_records.csv")
    st.markdown("**Dollar forecasts**")
    st.dataframe(pd.read_sql("SELECT * FROM dollar_forecasts ORDER BY id DESC", conn), use_container_width=True)
    st.markdown("**Weekly actuals**")
    weekly_all = pd.read_sql("SELECT * FROM weekly_actuals ORDER BY id DESC", conn)
    st.dataframe(weekly_all, use_container_width=True)
    st.download_button("Download weekly_actuals.csv", weekly_all.to_csv(index=False), "weekly_actuals.csv")
    st.markdown("**Sign-offs**")
    st.dataframe(pd.read_sql("SELECT * FROM signoffs ORDER BY id DESC", conn), use_container_width=True)
    st.caption("Download buttons above back up all data as CSV — recommended periodically, "
               "since free-tier hosting can reset local storage on redeploy.")
