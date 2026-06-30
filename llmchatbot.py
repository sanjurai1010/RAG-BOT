import os
import pandas as pd
import sqlite3
from groq import Groq
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json

# Page configuration
st.set_page_config(
    page_title="CSV Chatbot with Llama",
    page_icon="🦙",
    layout="wide"
)

# ============ CONFIGURATION ============
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    st.error('Error: GROQ_API_KEY environment variable not set. Please set it before running.')
    st.stop()

# ============ FUNCTIONS ============

def upload_csv_to_database(csv_file_path):
    df = pd.read_csv(csv_file_path)
    conn = sqlite3.connect('my_database.db')
    df.to_sql('data', conn, if_exists='replace', index=False)
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    conn.close()
    return columns, dtypes, df


def ask_question(question, columns, dtypes):
    """Convert natural language question to SQL using Llama via Groq"""
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""You are a SQL expert. Convert the following question to a SQLite query.

Database table name: data
Columns and their types: {json.dumps(dtypes)}

User question: {question}

Important: Return ONLY the SQL query without any explanation, markdown formatting, or additional text.
Just the raw SQL query that can be executed directly."""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a SQL expert. Generate only valid SQLite queries without any explanations or formatting."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
    )

    sql_query = chat_completion.choices[0].message.content.strip()
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    return sql_query


def decide_visualization(question, result_df, columns, dtypes):
    """
    Ask LLM whether a graph is appropriate and what kind to generate.
    Supports 16 chart types. Returns a dict or None.
    """
    client = Groq(api_key=GROQ_API_KEY)

    result_preview = result_df.head(30).to_dict(orient='records')
    result_columns = result_df.columns.tolist()
    result_dtypes = result_df.dtypes.astype(str).to_dict()

    prompt = f"""You are a data visualization expert. Your job is to decide the BEST chart for the user's query result.

User question: "{question}"

Query result columns: {result_columns}
Query result dtypes: {result_dtypes}
Query result data (up to 30 rows): {json.dumps(result_preview, default=str)}

Available chart types and when to use them:
- bar: comparing categories (e.g. sales by region)
- horizontal_bar: same as bar but horizontal, good for long category names
- line: trends over time or sequence
- area: trends over time with volume emphasis
- pie: part-to-whole with few categories (<=8)
- donut: same as pie but with a hole in center
- scatter: relationship between two numeric columns
- bubble: scatter with a third numeric dimension (size)
- histogram: distribution of a single numeric column
- box: statistical distribution, outliers, spread
- violin: like box but shows full distribution shape
- heatmap: correlation matrix or 2D frequency
- funnel: sequential stages with decreasing values
- treemap: hierarchical part-to-whole
- sunburst: hierarchical multi-level pie
- waterfall: running total, how values add/subtract

STRICT RULES:
1. If the user explicitly mentions any chart/graph/plot keyword, ALWAYS set should_visualize to true.
2. If data has 2+ rows and is numeric or categorical, prefer to visualize.
3. Only set should_visualize to false for a single scalar result like total count.
4. x_column and y_column MUST exactly match one of the result columns listed above.
5. For bubble chart, also fill size_column with a valid numeric column name.
6. For heatmap, x_column and y_column can be null (will use all numeric columns for correlation).
7. color_column should be null unless it meaningfully groups the data.

Respond ONLY with a valid JSON object, no explanation, no markdown:
{{
  "should_visualize": true or false,
  "chart_type": "<one of the chart types above or null>",
  "x_column": "<exact column name from result or null>",
  "y_column": "<exact column name from result or null>",
  "size_column": "<exact column name from result or null>",
  "color_column": "<exact column name from result or null>",
  "title": "<descriptive chart title>"
}}"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a data visualization expert. Always respond with valid JSON only. Never add explanation or markdown."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512,
    )

    raw = chat_completion.choices[0].message.content.strip()
    raw = raw.replace('```json', '').replace('```', '').strip()

    try:
        viz_config = json.loads(raw)
        return viz_config
    except Exception:
        return None


def generate_chart(viz_config, result_df):
    """Generate a Plotly chart based on LLM visualization config."""
    chart_type = viz_config.get("chart_type")
    x_col      = viz_config.get("x_column")
    y_col      = viz_config.get("y_column")
    size_col   = viz_config.get("size_column")
    color_col  = viz_config.get("color_column")
    title      = viz_config.get("title", "Chart")

    # Validate columns exist in result dataframe
    cols      = result_df.columns.tolist()
    x_col     = x_col     if x_col     in cols else None
    y_col     = y_col     if y_col     in cols else None
    size_col  = size_col  if size_col  in cols else None
    color_col = color_col if color_col in cols else None

    fig = None

    try:
        if chart_type == "bar":
            fig = px.bar(result_df, x=x_col, y=y_col, color=color_col,
                         title=title, text_auto=True)

        elif chart_type == "horizontal_bar":
            fig = px.bar(result_df, x=y_col, y=x_col, color=color_col,
                         title=title, orientation='h', text_auto=True)

        elif chart_type == "line":
            fig = px.line(result_df, x=x_col, y=y_col, color=color_col,
                          title=title, markers=True)

        elif chart_type == "area":
            fig = px.area(result_df, x=x_col, y=y_col, color=color_col, title=title)

        elif chart_type == "pie":
            fig = px.pie(result_df, names=x_col, values=y_col, title=title)

        elif chart_type == "donut":
            fig = px.pie(result_df, names=x_col, values=y_col, title=title, hole=0.4)

        elif chart_type == "scatter":
            fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col,
                             title=title)

        elif chart_type == "bubble":
            fig = px.scatter(result_df, x=x_col, y=y_col, size=size_col,
                             color=color_col, title=title)

        elif chart_type == "histogram":
            fig = px.histogram(result_df, x=x_col or y_col, color=color_col,
                               title=title, nbins=20)

        elif chart_type == "box":
            fig = px.box(result_df, x=x_col, y=y_col, color=color_col, title=title)

        elif chart_type == "violin":
            fig = px.violin(result_df, x=x_col, y=y_col, color=color_col,
                            title=title, box=True)

        elif chart_type == "heatmap":
            numeric_df = result_df.select_dtypes(include='number')
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig = px.imshow(corr, title=title, text_auto=True,
                                color_continuous_scale="RdBu_r", aspect="auto")
            elif x_col and y_col:
                pivot = result_df.pivot_table(index=y_col, columns=x_col,
                                              aggfunc='size', fill_value=0)
                fig = px.imshow(pivot, title=title, aspect="auto")

        elif chart_type == "funnel":
            fig = px.funnel(result_df, x=y_col, y=x_col, title=title)

        elif chart_type == "treemap":
            path_cols = [x_col] if x_col else [result_df.columns[0]]
            fig = px.treemap(result_df, path=path_cols, values=y_col, title=title)

        elif chart_type == "sunburst":
            path_cols = [x_col] if x_col else [result_df.columns[0]]
            fig = px.sunburst(result_df, path=path_cols, values=y_col, title=title)

        elif chart_type == "waterfall":
            if x_col and y_col:
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    x=result_df[x_col].tolist(),
                    y=result_df[y_col].tolist(),
                ))
                fig.update_layout(title=title)

        # Apply consistent dark theme
        if fig:
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=13),
                title_font_size=16,
            )

    except Exception as e:
        st.warning(f"⚠️ Could not render chart: {e}")
        fig = None

    return fig


def run_query(sql_query):
    conn = sqlite3.connect('my_database.db')
    result_df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return result_df


# ============ STREAMLIT UI ============

st.title("🦙 CSV Chatbot with Llama AI")
st.markdown("Upload your CSV and ask questions in plain English! Powered by **Llama 3.1 Instant** with 📊 **16 Chart Types**")

# Sidebar
with st.sidebar:
    st.header("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            columns, dtypes, df = upload_csv_to_database("temp.csv")
            st.session_state['columns'] = columns
            st.session_state['dtypes'] = dtypes

            st.success("✅ CSV uploaded successfully!")
            st.write(f"**Columns:** {', '.join(columns)}")
            st.write(f"**Total Rows:** {len(df)}")

            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    auto_viz = st.toggle("Auto-generate graphs when suitable", value=True)
    st.session_state['auto_viz'] = auto_viz

    st.markdown("---")
    st.subheader("📊 Supported Chart Types")
    st.markdown("""
    `bar` · `horizontal bar` · `line` · `area`
    `pie` · `donut` · `scatter` · `bubble`
    `histogram` · `box` · `violin` · `heatmap`
    `funnel` · `treemap` · `sunburst` · `waterfall`
    """)

# Main chat area
if uploaded_file is not None:
    st.header("💬 Ask Questions About Your Data")

    columns = st.session_state.get('columns', [])
    dtypes  = st.session_state.get('dtypes', {})

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sql" in message:
                st.code(message["sql"], language="sql")
            if "dataframe" in message and message["dataframe"] is not None:
                st.dataframe(message["dataframe"], use_container_width=True)
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)

    # Chat input
    user_question = st.chat_input("Ask anything... e.g. 'Show a donut chart of sales by category'")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                # Step 1: Generate SQL
                with st.spinner("🦙 Generating SQL query..."):
                    sql_query = ask_question(user_question, columns, dtypes)

                st.write("**Generated SQL Query:**")
                st.code(sql_query, language="sql")

                # Step 2: Execute SQL
                with st.spinner("⚙️ Running query..."):
                    result = run_query(sql_query)

                st.write(f"**Results:** ({len(result)} row(s))")
                st.dataframe(result, use_container_width=True)

                # Step 3: Decide & render visualization
                chart_fig = None

                # Detect if user is asking for a graph
                graph_keywords = [
                    "chart", "graph", "plot", "visualize", "show me",
                    "bar", "pie", "line", "scatter", "histogram", "donut",
                    "area", "bubble", "heatmap", "treemap", "sunburst",
                    "funnel", "violin", "box", "waterfall", "horizontal"
                ]
                user_wants_graph = any(kw in user_question.lower() for kw in graph_keywords)

                if st.session_state.get('auto_viz', True):
                    if user_wants_graph or len(result) > 1:
                        with st.spinner("📊 Deciding best visualization..."):
                            viz_config = decide_visualization(user_question, result, columns, dtypes)

                        if viz_config:
                            # Force visualization if user explicitly asked for a graph
                            if user_wants_graph:
                                viz_config["should_visualize"] = True

                            if viz_config.get("should_visualize"):
                                chart_fig = generate_chart(viz_config, result)
                                if chart_fig:
                                    chart_label = viz_config.get('chart_type', '').replace('_', ' ').capitalize()
                                    st.markdown(f"**📊 {chart_label} Chart:** *{viz_config.get('title', '')}*")
                                    st.plotly_chart(chart_fig, use_container_width=True)
                                else:
                                    st.info("ℹ️ Could not render chart for this data shape.")
                            else:
                                st.info("ℹ️ No chart needed for this result.")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Found {len(result)} result(s)",
                    "sql": sql_query,
                    "dataframe": result,
                    "chart": chart_fig
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}\n\nTry rephrasing your question."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Example questions
    st.markdown("---")
    st.subheader("💡 Example Questions:")

    col1, col2, col3 = st.columns(3)
    example_questions = [
        "Show a bar chart of count by department",
        "Plot a pie chart of sales by region",
        "Show a line chart of revenue over time",
        "Give me a donut chart of categories",
        "Show a scatter plot of age vs salary",
        "Show a heatmap of all correlations",
        "Plot a histogram of salary distribution",
        "Show a treemap of sales by category",
        "Give me a violin plot of age",
        "Show a funnel chart of stages",
        "Show a bubble chart of sales vs profit",
        "Plot a sunburst of region and category",
    ]

    for idx, example in enumerate(example_questions):
        col = [col1, col2, col3][idx % 3]
        with col:
            if st.button(example, key=f"example_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

else:
    st.info("👈 Please upload a CSV file from the sidebar to get started!")

    st.subheader("🚀 How to Use:")
    st.markdown("""
    1. Upload your CSV file from the sidebar
    2. Ask questions in plain English
    3. Ask for **any chart type** — the AI picks the right columns and renders it
    4. Supports **16 chart types**: bar, horizontal bar, line, area, pie, donut,
    scatter, bubble, histogram, box, violin, heatmap, funnel, treemap, sunburst, waterfall
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ Speed", "Ultra Fast")
    with col2:
        st.metric("📊 Chart Types", "16+")
    with col3:
        st.metric("🤖 Model", "Llama 3.1")
