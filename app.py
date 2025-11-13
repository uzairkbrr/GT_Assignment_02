import streamlit as st
import pandas as pd
from io import BytesIO
from scheduler import schedule_exams  # Make sure scheduler.py is in the same folder

st.set_page_config(page_title="FAST Peshawar Exam Scheduler", layout="wide")

st.title("📘 FAST Peshawar Sessional 1 Exam Scheduler")

st.markdown("""
This dashboard helps generate a **clash-free exam schedule**  
with constraints such as:
- ≤ 500 students per slot  
- 6 slots per day × 3 days  
- Minimum consecutive papers  
- Minimized 3+ papers in one day  
""")

# --- File Upload ---
import streamlit as st
import pandas as pd
from io import BytesIO
from scheduler import schedule_exams  # Make sure scheduler.py is in the same folder
import re

st.set_page_config(page_title="FAST Peshawar Exam Scheduler", layout="wide")

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
