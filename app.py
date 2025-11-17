import re
from io import BytesIO
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scheduler import schedule_exams

st.set_page_config(page_title="FAST Peshawar Exam Scheduler", layout="wide")

st.title("📘 FAST Peshawar Sessional 1 Exam Scheduler")

st.markdown(
    """
This dashboard helps generate a **clash-free exam schedule**  
while respecting the following constraints:
- ≤ 500 students per slot  
- 6 slots per day × 3 days  
- Minimum consecutive papers  
- Minimized 3+ papers in one day  
"""
)

# --- visualization helpers ---
def build_conflict_graph_figure(df: pd.DataFrame) -> go.Figure:
    """
    Construct a course-level conflict graph and return a Plotly figure.
    """
    df = df.dropna(subset=["Student ID", "Course Code", "Subject Name"])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Conflict Graph (no course data available)",
            xaxis={"visible": False},
            yaxis={"visible": False},
            plot_bgcolor="white",
        )
        return fig

    course_sets = df.groupby(["Course Code", "Subject Name"])["Student ID"].agg(lambda values: set(values))

    G = nx.Graph()
    for (code, subject), students in course_sets.items():
        node_id = (code, subject)
        G.add_node(node_id, code=code, subject=subject, size=len(students))

    course_items = list(course_sets.items())
    for idx, ((code_a, subject_a), students_a) in enumerate(course_items):
        for (code_b, subject_b), students_b in course_items[idx + 1 :]:
            shared = len(students_a & students_b)
            if shared:
                G.add_edge((code_a, subject_a), (code_b, subject_b), weight=shared)

    if not G.nodes:
        fig = go.Figure()
        fig.update_layout(
            title="Conflict Graph (no courses detected)",
            xaxis={"visible": False},
            yaxis={"visible": False},
            plot_bgcolor="white",
        )
        return fig

    pos = nx.spring_layout(G, seed=42)

    edge_x: List[float] = []
    edge_y: List[float] = []
    for source, target in G.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.8, color="#9aa5b1"),
        hoverinfo="none",
        mode="lines",
    )

    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_hover: List[str] = []
    node_size: List[float] = []
    node_color: List[int] = []

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        degree = len(list(G.neighbors(node)))
        node_x.append(x)
        node_y.append(y)
        node_text.append(data["code"])
        node_hover.append(
            f"{data['code']} – {data['subject']}<br>Students: {data['size']}<br>Conflicts: {degree}"
        )
        node_size.append(max(18, data["size"] / 3))
        node_color.append(degree)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Tealgrn",
            reversescale=True,
            showscale=True,
            colorbar=dict(title="Conflict degree"),
            line=dict(width=1.2, color="#1f2933"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Conflict Graph (Course-Level)",
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=40),
        height=720,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def build_timetable_heatmap(course_schedule: pd.DataFrame, students_per_slot: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap representing the timetable with hover details for each slot.
    """
    if course_schedule.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Timetable Heatmap (no scheduled courses)",
            xaxis={"visible": False},
            yaxis={"visible": False},
            plot_bgcolor="white",
        )
        return fig

    day_nums = sorted(course_schedule["DayNum"].dropna().unique())
    slot_nums = sorted(course_schedule["SlotNum"].dropna().unique())

    if not day_nums or not slot_nums:
        fig = go.Figure()
        fig.update_layout(
            title="Timetable Heatmap (slot metadata unavailable)",
            xaxis={"visible": False},
            yaxis={"visible": False},
            plot_bgcolor="white",
        )
        return fig

    day_labels = [f"Day {d}" for d in day_nums]
    slot_labels = [f"Slot {s}" for s in slot_nums]

    matrix = np.zeros((len(day_nums), len(slot_nums)))
    hover_text = [["No exams" for _ in slot_nums] for _ in day_nums]
    display_text = [["" for _ in slot_nums] for _ in day_nums]

    slot_student_map = students_per_slot.set_index(["DayNum", "SlotNum"])["Students"].to_dict()

    for i, day in enumerate(day_nums):
        for j, slot in enumerate(slot_nums):
            matrix[i, j] = slot_student_map.get((day, slot), 0)
            slot_courses = course_schedule[
                (course_schedule["DayNum"] == day) & (course_schedule["SlotNum"] == slot)
            ]
            if not slot_courses.empty:
                labels = [f"{row['Course Code']}" for _, row in slot_courses.iterrows()]
                subject_lines = [f"{row['Subject Name']}" for _, row in slot_courses.iterrows()]
                hover_lines = [f"{code} – {subject}" for code, subject in zip(labels, subject_lines)]
                hover_display = "<br>".join(hover_lines)
                hover_text[i][j] = hover_display

                max_inline = 5
                inline_courses = "<br>".join(labels[:max_inline])
                if len(labels) > max_inline:
                    inline_courses += "<br>…"
                display_text[i][j] = f"{int(matrix[i, j])} students<br>{len(labels)} papers<br>{inline_courses}"
            else:
                display_text[i][j] = "0 students<br>No papers"

    heatmap = go.Heatmap(
        z=matrix,
        x=slot_labels,
        y=day_labels,
        text=display_text,
        hovertext=hover_text,
        hovertemplate="%{y}, %{x}<br>Students: %{z:.0f}<br>%{hovertext}<extra></extra>",
        colorscale="Blues",
        colorbar=dict(title="Students"),
    )

    fig = go.Figure(data=[heatmap])
    fig.update_traces(texttemplate="%{text}", textfont=dict(color="#1f2933", size=11))
    fig.update_layout(
        title="Timetable Heatmap",
        xaxis=dict(title="Slot", side="top"),
        yaxis=dict(title="Day"),
        margin=dict(l=60, r=60, t=80, b=60),
        height=700,
        plot_bgcolor="white",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def render_fig_download(fig: go.Figure, label: str, filename: str) -> None:
    """
    Render a download button for a Plotly figure if static export is available.
    """
    try:
        image_bytes = fig.to_image(format="png", scale=2)
    except Exception:
        st.info("Install the 'kaleido' package to enable image downloads for charts.")
        return

    st.download_button(
        label,
        data=image_bytes,
        file_name=filename,
        mime="image/png",
    )

# --- helper: excel download ---
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    # Try xlsxwriter first, fall back to openpyxl
    engine = 'xlsxwriter'
    try:
        with pd.ExcelWriter(output, engine=engine) as writer:
            df.to_excel(writer, index=False)
    except Exception:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
    return output.getvalue()


# --- File Upload ---
uploaded_file = st.file_uploader("📂 Upload your Excel file (Student Data.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        df_orig = pd.read_excel(uploaded_file)
        st.success(f"✅ File uploaded successfully — {len(df_orig)} records found.")
        st.dataframe(df_orig.head(10))

        # Show a temporary "generating" message and clear it after schedule is ready
        placeholder = st.empty()
        placeholder.info("🔄 Generating schedule... please wait.")

        # Run Scheduler
        results = schedule_exams(df_orig)

        course_schedule = results["course_schedule"].copy()
        students_per_slot = results["students_per_slot"].copy()
        stats = results["stats"]

        # Clear the placeholder message now that we have results
        placeholder.empty()

        # --- Sort the final schedule by Day then Slot ---
        slot_re = re.compile(r"Day\s*(\d+)\s*-\s*Slot\s*(\d+)", re.IGNORECASE)

        def _parse_slot(slot_str):
            m = slot_re.search(str(slot_str))
            if m:
                return int(m.group(1)), int(m.group(2))
            return 0, 0

        course_schedule[['DayNum', 'SlotNum']] = course_schedule['Slot'].apply(lambda s: pd.Series(_parse_slot(s)))
        course_schedule = course_schedule.sort_values(['DayNum', 'SlotNum']).reset_index(drop=True)

        st.subheader("🗓️ Final Exam Schedule")
        st.dataframe(course_schedule.drop(columns=['DayNum', 'SlotNum']))

        # --- Students per Slot: create one card per day ---
        # parse students_per_slot to get day and slot
        students_per_slot[['DayNum', 'SlotNum']] = students_per_slot['Slot'].apply(lambda s: pd.Series(_parse_slot(s)))

        st.subheader("📊 Students per Slot (by Day)")
        day_cols = st.columns(3)
        for d in range(1, 4):
            with day_cols[d-1]:
                st.markdown(f"**Day {d}**")
                day_df = students_per_slot[students_per_slot['DayNum'] == d].copy()
                if day_df.empty:
                    st.info("No data for this day")
                else:
                    # order by SlotNum
                    day_df = day_df.sort_values('SlotNum')
                    # show bar chart with Slot label simplified
                    simple = day_df.copy()
                    simple['SlotLabel'] = simple['Slot'].apply(lambda s: s.split('-')[-1].strip())
                    st.bar_chart(simple.set_index('SlotLabel')['Students'])

        # --- Build student-level schedule by joining uploaded data to course_schedule
        # We need to make sure uploaded df has canonical column names; reuse a small mapping similar to scheduler
        def _normalize(name: str) -> str:
            s = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in str(name))
            return ' '.join(s.split())

        alias_map = {
            "Student ID": ["roll no", "rollno", "reg no", "regno", "student no", "studentid", "id"],
            "Course Code": ["code", "coursecode", "course code", "course_code", "code "],
            "Section": ["section", "sec"],
            "Subject Name": ["course", "subject", "subject name", "course name", "course_name"]
        }

        normalized_columns = { _normalize(c): c for c in df_orig.columns }
        alias_lookup = {}
        for canon, aliases in alias_map.items():
            for a in aliases:
                alias_lookup[_normalize(a)] = canon

        matched_columns = {}
        import difflib
        for req in ["Student ID", "Course Code", "Section", "Subject Name"]:
            norm = _normalize(req)
            if norm in normalized_columns:
                matched_columns[req] = normalized_columns[norm]
                continue
            found = False
            for col_norm, orig_col in normalized_columns.items():
                if alias_lookup.get(col_norm) == req:
                    matched_columns[req] = orig_col
                    found = True
                    break
            if found:
                continue
            candidates = difflib.get_close_matches(norm, list(normalized_columns.keys()), n=1, cutoff=0.75)
            if candidates:
                matched_columns[req] = normalized_columns[candidates[0]]

        # If mapping incomplete, try to proceed but warn
        missing = [r for r in ["Student ID", "Course Code", "Subject Name"] if r not in matched_columns]
        if missing:
            st.warning(f"Could not auto-detect columns: {', '.join(missing)}. Student-level details may be incomplete.")

        df = df_orig.rename(columns={matched_columns[k]: k for k in matched_columns})

        # Merge to get student's assigned Slot
        student_schedule = df.merge(course_schedule[['Course Code', 'Subject Name', 'Slot']], on=['Course Code', 'Subject Name'], how='left')

        # Normalize slot into DayNum and SlotNum on student_schedule
        student_schedule[['DayNum', 'SlotNum']] = student_schedule['Slot'].apply(lambda s: pd.Series(_parse_slot(s)))

        # --- Compute statistics datasets: 3_per_day, 4_per_day ---
        per_student_day = student_schedule.groupby(['Student ID', 'DayNum']).agg(
            Papers=('Course Code', 'nunique')
        ).reset_index()

        three_per_day_keys = per_student_day[per_student_day['Papers'] == 3][['Student ID', 'DayNum']]
        four_per_day_keys = per_student_day[per_student_day['Papers'] == 4][['Student ID', 'DayNum']]

        def records_for_keys(keys_df):
            if keys_df.empty:
                return pd.DataFrame(columns=student_schedule.columns)
            merged = keys_df.merge(student_schedule, on=['Student ID', 'DayNum'], how='left')
            # keep relevant columns and sort
            return merged[['Student ID', 'Course Code', 'Subject Name', 'Slot', 'DayNum', 'SlotNum']].sort_values(['Student ID', 'DayNum', 'SlotNum'])

        records_3_per_day = records_for_keys(three_per_day_keys)
        records_4_per_day = records_for_keys(four_per_day_keys)

        # --- Compute consecutive papers per day (3 and 4 consecutive slots)
        consec_records_3 = []
        consec_records_4 = []

        for (sid, day), group in student_schedule.groupby(['Student ID', 'DayNum']):
            slots = sorted(group['SlotNum'].dropna().astype(int).unique())
            # find runs
            run = [slots[0]] if slots else []
            runs = []
            for sn in slots[1:]:
                if sn == run[-1] + 1:
                    run.append(sn)
                else:
                    runs.append(run)
                    run = [sn]
            if run:
                runs.append(run)
            # check runs for length >=3 or >=4
            for r in runs:
                if len(r) >= 3:
                    recs = group[group['SlotNum'].isin(r)][['Student ID', 'Course Code', 'Subject Name', 'Slot', 'DayNum', 'SlotNum']]
                    consec_records_3.append(recs)
                if len(r) >= 4:
                    recs = group[group['SlotNum'].isin(r)][['Student ID', 'Course Code', 'Subject Name', 'Slot', 'DayNum', 'SlotNum']]
                    consec_records_4.append(recs)

        records_3_consecutive = pd.concat(consec_records_3, ignore_index=True) if consec_records_3 else pd.DataFrame(columns=['Student ID', 'Course Code', 'Subject Name', 'Slot', 'DayNum', 'SlotNum'])
        records_4_consecutive = pd.concat(consec_records_4, ignore_index=True) if consec_records_4 else pd.DataFrame(columns=['Student ID', 'Course Code', 'Subject Name', 'Slot', 'DayNum', 'SlotNum'])

        # --- Summary Statistics display ---
        st.subheader("📈 Summary Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Students with 3 papers/day", stats.get("3_per_day", len(records_3_per_day['Student ID'].unique())))
            st.metric("Students with 4 papers/day", stats.get("4_per_day", len(records_4_per_day['Student ID'].unique())))
        with col2:
            st.metric("Students with 3 consecutive papers", stats.get("3_consecutive", len(records_3_consecutive['Student ID'].unique())))
            st.metric("Students with 4 consecutive papers", stats.get("4_consecutive", len(records_4_consecutive['Student ID'].unique())))

        # Show detailed records in expanders
        with st.expander("Show students with 3 papers in a single day"):
            if records_3_per_day.empty:
                st.write("No students found with exactly 3 papers in a day.")
            else:
                st.dataframe(records_3_per_day)

        with st.expander("Show students with 4 papers in a single day"):
            if records_4_per_day.empty:
                st.write("No students found with exactly 4 papers in a day.")
            else:
                st.dataframe(records_4_per_day)

        with st.expander("Show students with 3 consecutive papers"):
            if records_3_consecutive.empty:
                st.write("No students found with 3 consecutive papers.")
            else:
                st.dataframe(records_3_consecutive)

        with st.expander("Show students with 4 consecutive papers"):
            if records_4_consecutive.empty:
                st.write("No students found with 4 consecutive papers.")
            else:
                st.dataframe(records_4_consecutive)

        # --- Graph visualisations ---
        st.subheader("🔗 Conflict Graph (Course-Level)")
        conflict_fig = build_conflict_graph_figure(df)
        st.plotly_chart(conflict_fig, use_container_width=True)
        render_fig_download(conflict_fig, "Download Conflict Graph (PNG)", "conflict_graph.png")

        st.subheader("🖼️ Timetable Heatmap")
        timetable_fig = build_timetable_heatmap(course_schedule, students_per_slot)
        st.plotly_chart(timetable_fig, use_container_width=True)
        render_fig_download(timetable_fig, "Download Timetable Heatmap (PNG)", "timetable_heatmap.png")

        # --- Download buttons for all outputs ---
        st.subheader("⬇️ Download generated files")
        cola, colb = st.columns(2)
        with cola:
            st.download_button("Download Full Schedule (Excel)", data=to_excel_bytes(course_schedule.drop(columns=['DayNum', 'SlotNum'])), file_name="Exam_Schedule.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download Students per Slot (Excel)", data=to_excel_bytes(students_per_slot.drop(columns=['DayNum', 'SlotNum'])), file_name="Students_per_Slot.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download Student Schedule (Excel)", data=to_excel_bytes(student_schedule), file_name="Student_Schedule.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with colb:
            st.download_button("Download 3_per_day records", data=to_excel_bytes(records_3_per_day), file_name="3_per_day_records.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.download_button("Download 4_per_day records", data=to_excel_bytes(records_4_per_day), file_name="4_per_day_records.xlsx", mime="application/vnd.openxmlformats-officedocument-spreadsheetml.sheet")
            st.download_button("Download 3_consecutive records", data=to_excel_bytes(records_3_consecutive), file_name="3_consecutive_records.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("👆 Please upload the student registration Excel file to continue.")
