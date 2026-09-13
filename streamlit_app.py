
import os
from datetime import date

import pandas as pd
import streamlit as st

try:
    from flask_app.data_service import predict_employee, get_task_options, DATA_FILE
except Exception:
    predict_employee = None
    get_task_options = None
    DATA_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "raw",
        "employee_tasks_dataset_3.csv",
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREATED_TASKS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "created_tasks.csv"
)
LOGO_PATH = os.path.join(
    BASE_DIR,
    "flask_app",
    "static",
    "css",
    "images",
    "synq-logo.png",
)

st.set_page_config(
    page_title="SYNQ",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# GLOBAL STYLE — MATCHES FLASK SYNQ DARK UI
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #050505;
        --panel: #0b0b0d;
        --panel-2: #101012;
        --border: #242424;
        --muted: #7f7f86;
        --text: #ffffff;
        --purple: #a855f7;
        --purple-2: #7c3aed;
        --green: #22c55e;
        --red: #ef4444;
        --yellow: #eab308;
        --blue: #3b82f6;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 8%, rgba(124,58,237,.08), transparent 22%),
            radial-gradient(circle at 35% 100%, rgba(168,85,247,.05), transparent 28%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        width: 270px !important;
        min-width: 270px !important;
        background:
            radial-gradient(circle at 20% 10%, rgba(116,37,190,.08), transparent 25%),
            #070707;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 24px;
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #555 !important;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 4px;
    }

    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
        min-height: 48px;
        border-radius: 11px;
        padding: 0 14px;
        color: #8a8a90;
        transition: .2s ease;
    }

    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
        background: #100b16;
        color: #fff;
    }

    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    [data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(
            90deg,
            rgba(143,52,255,.20),
            rgba(143,52,255,.05)
        );
        color: #fff;
        border: 1px solid rgba(168,85,247,.18);
    }

    .block-container {
        max-width: 1500px;
        padding: 36px 44px 60px 44px;
    }

    h1, h2, h3, p, label {
        color: #fff;
    }

    .synq-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 6px 24px 4px;
    }

    .synq-brand-title {
        font-size: 29px;
        letter-spacing: 5px;
        font-weight: 800;
        line-height: 1;
    }

    .synq-brand-sub {
        margin-top: 7px;
        color: #747474;
        font-size: 8px;
        letter-spacing: 1.7px;
    }

    .page-header {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:20px;
        margin-bottom:24px;
    }

    .page-title {
        font-size: 32px;
        font-weight: 780;
        letter-spacing: -.7px;
        margin-bottom: 5px;
    }

    .page-subtitle {
        color: #777;
        font-size: 14px;
    }

    .status-chip {
        display:inline-flex;
        align-items:center;
        gap:8px;
        color:#bcbcbc;
        font-size:12px;
        background:#0d0d0f;
        border:1px solid #242424;
        padding:10px 13px;
        border-radius:10px;
    }

    .status-dot {
        width:7px;
        height:7px;
        border-radius:50%;
        background:#22c55e;
        box-shadow:0 0 10px rgba(34,197,94,.65);
    }

    .hero {
        position: relative;
        overflow: hidden;
        min-height: 210px;
        border:1px solid #232323;
        border-radius:18px;
        padding:32px;
        margin: 8px 0 25px;
        background:
            linear-gradient(125deg, rgba(168,85,247,.12), rgba(7,7,7,.2) 55%),
            #0a0a0b;
    }

    .hero:after {
        content:"";
        position:absolute;
        width:270px;
        height:270px;
        right:-80px;
        top:-100px;
        border-radius:50%;
        background:rgba(168,85,247,.10);
        filter:blur(10px);
    }

    .hero-label {
        color:#a855f7;
        font-size:10px;
        letter-spacing:2.2px;
        font-weight:700;
        margin-bottom:13px;
    }

    .hero h2 {
        font-size:35px;
        margin:0 0 10px;
        letter-spacing:-1px;
    }

    .hero h2 span {
        color:#a855f7;
    }

    .hero p {
        max-width:720px;
        color:#939393;
        font-size:14px;
        line-height:1.75;
    }

    .metric-card {
        min-height: 132px;
        background: linear-gradient(145deg,#0d0d0f,#09090a);
        border:1px solid #232323;
        border-radius:15px;
        padding:20px;
    }

    .metric-title {
        color:#747474;
        font-size:10px;
        letter-spacing:1.8px;
        font-weight:700;
    }

    .metric-value {
        margin-top:13px;
        font-size:28px;
        font-weight:780;
        color:#fff;
    }

    .metric-sub {
        margin-top:6px;
        color:#676767;
        font-size:11px;
    }

    .panel {
        background:#0b0b0d;
        border:1px solid #232323;
        border-radius:16px;
        padding:21px;
        margin-top:14px;
    }

    .panel-title {
        font-size:16px;
        font-weight:700;
        margin-bottom:3px;
    }

    .panel-sub {
        color:#6e6e73;
        font-size:11px;
        margin-bottom:16px;
    }

    .result-card {
        background:
            linear-gradient(140deg,rgba(168,85,247,.09),transparent 58%),
            #0b0b0d;
        border:1px solid #2b2333;
        border-radius:17px;
        padding:24px;
        margin-top:22px;
    }

    .result-label {
        color:#7b7b82;
        font-size:10px;
        letter-spacing:1.6px;
        text-transform:uppercase;
    }

    .result-value {
        margin-top:7px;
        font-size:24px;
        font-weight:750;
    }

    .pill {
        display:inline-block;
        border:1px solid #2b2b2f;
        background:#111113;
        padding:6px 9px;
        border-radius:99px;
        color:#aaa;
        font-size:10px;
    }

    div[data-testid="stForm"] {
        background:#0b0b0d;
        border:1px solid #232323;
        border-radius:16px;
        padding:22px;
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    div[data-baseweb="select"] > div {
        background:#0f0f11 !important;
        border-color:#2a2a2d !important;
        color:#fff !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background:linear-gradient(135deg,#9333ea,#7c3aed);
        border:1px solid #a855f7;
        color:#fff;
        border-radius:10px;
        min-height:42px;
        font-weight:700;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border-color:#c084fc;
        color:#fff;
    }

    [data-testid="stDataFrame"] {
        border:1px solid #242424;
        border-radius:14px;
        overflow:hidden;
    }

    [data-testid="stMetric"] {
        background:#0b0b0d;
        border:1px solid #232323;
        border-radius:14px;
        padding:16px;
    }

    hr {
        border-color:#222 !important;
    }

    @media(max-width:900px){
        .block-container{padding:25px 18px 50px}
        .page-header{flex-direction:column}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATA HELPERS
# =========================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE)

    for col in [
        "rating",
        "error_risk",
        "is_completed",
        "days_to_deadline",
        "volume_metric",
        "dependency_score",
        "perceived_difficulty",
        "primary_skill_matching",
        "secondary_skill_matching",
        "ternary_skill_matching",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_created_tasks():
    if not os.path.exists(CREATED_TASKS_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(CREATED_TASKS_FILE)
    except Exception:
        return pd.DataFrame()


def save_created_tasks(tasks_df):
    os.makedirs(os.path.dirname(CREATED_TASKS_FILE), exist_ok=True)
    tasks_df.to_csv(CREATED_TASKS_FILE, index=False)


def task_options(df):
    try:
        if get_task_options:
            options = get_task_options()
            if isinstance(options, dict):
                return {
                    "task_types": options.get("task_types", []),
                    "priorities": options.get("priorities", []),
                }
    except Exception:
        pass

    task_types = (
        sorted(df["task_type"].dropna().astype(str).unique().tolist())
        if "task_type" in df.columns
        else []
    )
    priorities = (
        sorted(df["priority"].dropna().astype(str).unique().tolist())
        if "priority" in df.columns
        else ["Low", "Medium", "High"]
    )
    return {"task_types": task_types, "priorities": priorities}


def page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="page-header">
          <div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
          </div>
          <div class="status-chip">
            <span class="status-dot"></span>
            SYNQ Intelligence Active
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title, value, subtitle):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def difficulty_to_number(value):
    mapping = {
        "Low": 1.0,
        "Easy": 1.0,
        "Medium": 3.0,
        "Moderate": 3.0,
        "High": 5.0,
        "Hard": 5.0,
    }
    return mapping.get(str(value), 3.0)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
    else:
        st.markdown(
            """
            <div class="synq-brand">
                <div>
                    <div class="synq-brand-title">SYNQ</div>
                    <div class="synq-brand-sub">INTELLIGENT WORKPLACE</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    page = st.radio(
        "Workspace",
        [
            "◆  Dashboard",
            "◎  Predict Employee",
            "＋  Create Task",
            "▣  Tasks",
            "♟  Employee Data",
            "▮  Analytics",
            "✦  Insights",
        ],
        label_visibility="visible",
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("SYNQ • AI Workforce Intelligence")


df = load_dataset()

if df.empty:
    st.error(
        "Dataset not found. Expected: "
        "`data/raw/employee_tasks_dataset_3.csv`"
    )
    st.stop()

options = task_options(df)


# =========================================================
# DASHBOARD
# =========================================================

if page.endswith("Dashboard"):
    page_header(
        "Dashboard",
        "Employee performance intelligence overview",
    )

    st.markdown(
        """
        <section class="hero">
            <div class="hero-label">AI-POWERED WORKFORCE INTELLIGENCE</div>
            <h2>Welcome to <span>SYNQ.</span></h2>
            <p>
                Monitor employee performance, evaluate task risk,
                identify high-performing employees and make smarter
                workforce decisions using data-driven intelligence.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    total_employees = int(df["employee_id"].nunique())
    total_tasks = int(len(df))
    avg_rating = float(df["rating"].mean()) if "rating" in df else 0
    avg_error = float(df["error_risk"].mean()) if "error_risk" in df else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("TOTAL EMPLOYEES", f"{total_employees:,}", "Active workforce")
    with c2:
        metric_card("TOTAL TASKS", f"{total_tasks:,}", "Historical task records")
    with c3:
        metric_card("AVG RATING", f"{avg_rating:.2f}", "Employee performance")
    with c4:
        metric_card("ERROR RISK", f"{avg_error:.2f}", "Overall risk level")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown(
            """
            <div class="panel-title" style="margin-top:24px">Quick Overview</div>
            <div class="panel-sub">Latest workforce indicators</div>
            """,
            unsafe_allow_html=True,
        )

        if "task_type" in df.columns:
            task_counts = df["task_type"].value_counts().head(8)
            st.bar_chart(task_counts)

    with right:
        st.markdown(
            """
            <div class="panel-title" style="margin-top:24px">Priority Mix</div>
            <div class="panel-sub">Distribution of assigned priorities</div>
            """,
            unsafe_allow_html=True,
        )

        if "priority" in df.columns:
            priority_counts = df["priority"].value_counts()
            st.bar_chart(priority_counts)


# =========================================================
# PREDICT EMPLOYEE
# =========================================================

elif page.endswith("Predict Employee"):
    page_header(
        "Predict Employee",
        "Find the best-fit employee using trained SYNQ ML models",
    )

    with st.form("predict_form"):
        c1, c2 = st.columns(2)

        with c1:
            task_type = st.selectbox(
                "Task Type",
                options["task_types"] or ["Feature Development"],
            )
            priority = st.selectbox(
                "Priority",
                options["priorities"] or ["Low", "Medium", "High"],
            )
            days_to_deadline = st.number_input(
                "Days to Deadline",
                min_value=0.0,
                value=7.0,
                step=1.0,
            )

        with c2:
            volume_metric = st.number_input(
                "Volume Metric",
                min_value=0.0,
                value=10.0,
                step=1.0,
            )
            perceived_difficulty = st.selectbox(
                "Perceived Difficulty",
                ["Low", "Medium", "High"],
                index=1,
            )
            error_risk = st.number_input(
                "Error Risk",
                min_value=0.0,
                value=2.0,
                step=0.1,
            )

        dependency_score = st.slider(
            "Dependency Score",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
        )

        submitted = st.form_submit_button(
            "Run SYNQ Prediction",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "task_type": task_type,
            "priority": priority,
            "days_to_deadline": float(days_to_deadline),
            "volume_metric": float(volume_metric),
            "perceived_difficulty": difficulty_to_number(perceived_difficulty),
            "error_risk": float(error_risk),
            "dependency_score": float(dependency_score),
        }

        if predict_employee is None:
            st.error(
                "Could not import `predict_employee` from "
                "`flask_app/data_service.py`."
            )
        else:
            try:
                with st.spinner("SYNQ is evaluating employee models..."):
                    result = predict_employee(payload)

                employee = result.get("best_fit_employee", "N/A")
                rating = result.get("predicted_rating", "N/A")
                category = result.get(
                    "category",
                    result.get("performance", "N/A"),
                )
                risk = result.get(
                    "risk_level",
                    result.get("risk", "N/A"),
                )
                model_type = result.get("model_type", "ML Model")
                fallback = bool(result.get("fallback_used", False))

                st.markdown(
                    '<div class="result-card">',
                    unsafe_allow_html=True,
                )

                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.metric("Best Fit Employee", employee)
                with r2:
                    try:
                        st.metric("Predicted Rating", f"{float(rating):.2f}/10")
                    except Exception:
                        st.metric("Predicted Rating", str(rating))
                with r3:
                    st.metric("Performance", str(category))
                with r4:
                    st.metric("Risk Level", str(risk))

                if fallback:
                    st.warning("Historical dataset fallback used")
                else:
                    st.success(f"Prediction generated using: {model_type}")

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# =========================================================
# CREATE TASK
# =========================================================

elif page.endswith("Create Task"):
    page_header(
        "Create Task",
        "Add a new task to the SYNQ task workspace",
    )

    with st.form("create_task_form", clear_on_submit=True):
        c1, c2 = st.columns(2)

        with c1:
            task_type = st.selectbox(
                "Task Type",
                options["task_types"] or ["Feature Development"],
                key="create_type",
            )
            priority = st.selectbox(
                "Priority",
                options["priorities"] or ["Low", "Medium", "High"],
                key="create_priority",
            )
            days_to_deadline = st.number_input(
                "Days to Deadline",
                min_value=0,
                value=7,
                step=1,
            )
            task_date = st.date_input(
                "Task Date",
                value=date.today(),
            )

        with c2:
            volume_metric = st.number_input(
                "Volume Metric",
                min_value=0.0,
                value=10.0,
                step=1.0,
            )
            perceived_difficulty = st.selectbox(
                "Perceived Difficulty",
                ["Low", "Medium", "High"],
                index=1,
            )
            error_risk = st.number_input(
                "Error Risk",
                min_value=0.0,
                value=2.0,
                step=0.1,
            )
            dependency_score = st.number_input(
                "Dependency Score",
                min_value=0.0,
                value=2.0,
                step=0.1,
            )

        create_clicked = st.form_submit_button(
            "Create Task",
            use_container_width=True,
        )

    if create_clicked:
        tasks_df = load_created_tasks()

        next_num = 1
        if not tasks_df.empty and "task_id" in tasks_df.columns:
            ids = (
                tasks_df["task_id"]
                .astype(str)
                .str.extract(r"(\d+)$")[0]
            )
            ids = pd.to_numeric(ids, errors="coerce").dropna()
            if not ids.empty:
                next_num = int(ids.max()) + 1

        new_task = {
            "task_id": f"TSK-{next_num:04d}",
            "task_type": task_type,
            "priority": priority,
            "days_to_deadline": float(days_to_deadline),
            "volume_metric": float(volume_metric),
            "perceived_difficulty": difficulty_to_number(perceived_difficulty),
            "error_risk": float(error_risk),
            "dependency_score": float(dependency_score),
            "task_date": task_date.isoformat(),
        }

        tasks_df = pd.concat(
            [tasks_df, pd.DataFrame([new_task])],
            ignore_index=True,
        )
        save_created_tasks(tasks_df)
        st.success(f"Task {new_task['task_id']} created successfully.")


# =========================================================
# TASKS
# =========================================================

elif page.endswith("Tasks"):
    page_header(
        "Tasks",
        "View, edit and delete tasks created in SYNQ",
    )

    tasks_df = load_created_tasks()

    if tasks_df.empty:
        st.info("No created tasks yet.")
    else:
        st.dataframe(
            tasks_df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.subheader("Edit / Delete Task")

        task_ids = tasks_df["task_id"].astype(str).tolist()
        selected_id = st.selectbox("Select Task", task_ids)

        selected_row = tasks_df[
            tasks_df["task_id"].astype(str) == selected_id
        ].iloc[0]

        ec1, ec2 = st.columns(2)

        with ec1:
            edit_priority = st.selectbox(
                "Edit Priority",
                options["priorities"] or ["Low", "Medium", "High"],
                index=(
                    (options["priorities"] or ["Low", "Medium", "High"]).index(
                        str(selected_row.get("priority"))
                    )
                    if str(selected_row.get("priority"))
                    in (options["priorities"] or ["Low", "Medium", "High"])
                    else 0
                ),
            )

        with ec2:
            edit_deadline = st.number_input(
                "Edit Days to Deadline",
                min_value=0.0,
                value=float(selected_row.get("days_to_deadline", 0)),
            )

        b1, b2 = st.columns(2)

        with b1:
            if st.button(
                "Save Task Changes",
                use_container_width=True,
            ):
                mask = tasks_df["task_id"].astype(str) == selected_id
                tasks_df.loc[mask, "priority"] = edit_priority
                tasks_df.loc[mask, "days_to_deadline"] = float(edit_deadline)
                save_created_tasks(tasks_df)
                st.success(f"{selected_id} updated.")
                st.rerun()

        with b2:
            if st.button(
                "Delete Task",
                use_container_width=True,
            ):
                tasks_df = tasks_df[
                    tasks_df["task_id"].astype(str) != selected_id
                ]
                save_created_tasks(tasks_df)
                st.success(f"{selected_id} deleted.")
                st.rerun()


# =========================================================
# EMPLOYEE DATA
# =========================================================

elif page.endswith("Employee Data"):
    page_header(
        "Employee Data",
        "Explore workforce performance records",
    )

    employee_summary = (
        df.groupby("employee_id")
        .agg(
            total_tasks=("task_id", "count"),
            avg_rating=("rating", "mean"),
            avg_error_risk=("error_risk", "mean"),
        )
        .reset_index()
    )

    if "is_completed" in df.columns:
        completed = (
            df.groupby("employee_id")["is_completed"]
            .mean()
            .mul(100)
            .round(2)
            .rename("completion_rate")
            .reset_index()
        )
        employee_summary = employee_summary.merge(
            completed,
            on="employee_id",
            how="left",
        )

    employee_summary["avg_rating"] = employee_summary["avg_rating"].round(2)
    employee_summary["avg_error_risk"] = employee_summary[
        "avg_error_risk"
    ].round(2)

    search = st.text_input(
        "Search Employee",
        placeholder="EMP_001",
    )

    display_df = employee_summary.copy()
    if search.strip():
        display_df = display_df[
            display_df["employee_id"]
            .astype(str)
            .str.contains(search.strip(), case=False, na=False)
        ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# ANALYTICS
# =========================================================

elif page.endswith("Analytics"):
    page_header(
        "Analytics",
        "Workforce trends, performance distribution and task intelligence",
    )

    completion_rate = (
        float(df["is_completed"].mean() * 100)
        if "is_completed" in df.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Employees", int(df["employee_id"].nunique()))
    with c2:
        st.metric("Tasks", f"{len(df):,}")
    with c3:
        st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
    with c4:
        st.metric("Completion", f"{completion_rate:.1f}%")

    a1, a2 = st.columns(2)

    with a1:
        st.subheader("Performance Distribution")
        bins = [-float("inf"), 4, 6, 8, float("inf")]
        labels = [
            "Needs Improvement",
            "Average Performer",
            "Good Performer",
            "High Performer",
        ]
        perf = pd.cut(
            df["rating"],
            bins=bins,
            labels=labels,
            right=False,
        ).value_counts().reindex(labels, fill_value=0)
        st.bar_chart(perf)

    with a2:
        st.subheader("Task Type Distribution")
        st.bar_chart(df["task_type"].value_counts())

    a3, a4 = st.columns(2)

    with a3:
        st.subheader("Priority Distribution")
        st.bar_chart(df["priority"].value_counts())

    with a4:
        st.subheader("Error Risk by Priority")
        risk_by_priority = (
            df.groupby("priority")["error_risk"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(risk_by_priority)

    if "task_given_date" in df.columns:
        temp = df.copy()
        temp["task_given_date"] = pd.to_datetime(
            temp["task_given_date"],
            errors="coerce",
        )
        trend = (
            temp.dropna(subset=["task_given_date"])
            .set_index("task_given_date")
            .resample("ME")["rating"]
            .mean()
        )
        if not trend.empty:
            st.subheader("Performance Trend")
            st.line_chart(trend)


# =========================================================
# INSIGHTS
# =========================================================

elif page.endswith("Insights"):
    page_header(
        "Insights",
        "Automatically generated workforce intelligence",
    )

    employee_perf = (
        df.groupby("employee_id")["rating"]
        .mean()
        .sort_values(ascending=False)
    )

    best_employee = employee_perf.index[0]
    best_rating = float(employee_perf.iloc[0])

    task_perf = (
        df.groupby("task_type")["rating"]
        .mean()
        .sort_values(ascending=False)
    )

    best_task = task_perf.index[0]
    best_task_rating = float(task_perf.iloc[0])

    avg_error = float(df["error_risk"].mean())

    i1, i2 = st.columns(2)

    with i1:
        st.success(
            f"Top performer: {best_employee} "
            f"with an average rating of {best_rating:.2f}."
        )

        st.info(
            f"Strongest task category: {best_task} "
            f"with an average rating of {best_task_rating:.2f}."
        )

    with i2:
        if avg_error >= 4:
            st.error(
                f"Overall error risk is elevated at {avg_error:.2f}."
            )
        elif avg_error >= 2:
            st.warning(
                f"Overall error risk is moderate at {avg_error:.2f}."
            )
        else:
            st.success(
                f"Overall error risk is low at {avg_error:.2f}."
            )

        if "is_completed" in df.columns:
            completion = float(df["is_completed"].mean() * 100)
            st.info(
                f"Overall completion rate is {completion:.1f}%."
            )

    st.subheader("Employee Performance Ranking")
    ranking = employee_perf.round(2).rename("Average Rating").reset_index()
    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True,
    )
