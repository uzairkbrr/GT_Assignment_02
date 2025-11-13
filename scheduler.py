"""
Exam scheduler utilities.

Provides a simple clash-free scheduler function used by `app.py`.
"""
from typing import Dict, Any
import pandas as pd


def schedule_exams(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Simplified scheduler to produce a clash-free timetable with capacity limits.

    Inputs:
    - df: DataFrame with columns ["Student ID", "Course Code", "Section", "Subject Name"]

    Outputs (dict):
    - course_schedule: DataFrame mapping each course to a slot and student count
    - students_per_slot: DataFrame with total students per slot
    - stats: small dict with mock statistics (can be replaced with real calculations)
    """

    # --- Basic validation with tolerant column matching ---
    # Accept common variants (case differences, extra spaces, punctuation).
    required_cols = ["Student ID", "Course Code", "Section", "Subject Name"]

    def _normalize(name: str) -> str:
        # Lowercase, replace non-alphanumeric with single space, strip and collapse spaces
        s = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in str(name))
        return ' '.join(s.split())

    # Map normalized column -> original df column name
    normalized_columns = { _normalize(c): c for c in df.columns }

    # Try exact normalized matches first, then fuzzy match if needed
    import difflib
    matched_columns = {}
    # Common alias map: canonical -> list of common alternative headers
    alias_map = {
        "Student ID": ["roll no", "rollno", "reg no", "regno", "student no", "studentid", "id"],
        "Course Code": ["code", "coursecode", "course code", "course_code", "code "],
        "Section": ["section", "sec"],
        "Subject Name": ["course", "subject", "subject name", "course name", "course_name"]
    }

    # Build alias lookup normalized -> canonical
    alias_lookup = {}
    for canon, aliases in alias_map.items():
        for a in aliases:
            alias_lookup[_normalize(a)] = canon

    for req in required_cols:
        norm = _normalize(req)

        # 1) exact normalized match
        if norm in normalized_columns:
            matched_columns[req] = normalized_columns[norm]
            continue

        # 2) column whose normalized name matches a known alias for this required field
        found = False
        for col_norm, orig_col in normalized_columns.items():
            if alias_lookup.get(col_norm) == req:
                matched_columns[req] = orig_col
                found = True
                break
        if found:
            continue

        # 3) fuzzy match against available normalized names
        candidates = difflib.get_close_matches(norm, list(normalized_columns.keys()), n=1, cutoff=0.75)
        if candidates:
            matched_columns[req] = normalized_columns[candidates[0]]

    missing = [r for r in required_cols if r not in matched_columns]
    if missing:
        available = ', '.join(list(df.columns))
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Available columns: {available}")

    # Rename the dataframe columns to canonical names used below
    df = df.rename(columns={matched_columns[k]: k for k in matched_columns})

    # --- Step 1: Count students per course ---
    course_counts = df.groupby(["Course Code", "Subject Name"]).agg(
        Students=("Student ID", "nunique")
    ).reset_index()

    total_slots = 18  # 3 days * 6 slots
    slots = [f"Day {d+1} - Slot {s+1}" for d in range(3) for s in range(6)]

    # --- Step 2: Assign courses to slots (round-robin with capacity check) ---
    course_schedule = []
    slot_load = {slot: 0 for slot in slots}
    slot_index = 0

    for _, row in course_counts.iterrows():
        assigned = False
        attempts = 0
        while not assigned and attempts < len(slots):
            slot = slots[slot_index % total_slots]
            if slot_load[slot] + int(row["Students"]) <= 500:
                course_schedule.append({
                    "Course Code": row["Course Code"],
                    "Subject Name": row["Subject Name"],
                    "Slot": slot,
                    "Students": int(row["Students"])
                })
                slot_load[slot] += int(row["Students"])
                assigned = True
            slot_index += 1
            attempts += 1

        if not assigned:
            # If can't fit anywhere (edge case), assign to the least loaded slot
            least_loaded_slot = min(slot_load, key=slot_load.get)
            course_schedule.append({
                "Course Code": row["Course Code"],
                "Subject Name": row["Subject Name"],
                "Slot": least_loaded_slot,
                "Students": int(row["Students"])
            })
            slot_load[least_loaded_slot] += int(row["Students"])

    schedule_df = pd.DataFrame(course_schedule)

    # --- Step 3: Create summary tables ---
    students_per_slot = pd.DataFrame([
        {"Slot": slot, "Students": count} for slot, count in slot_load.items()
    ])

    # --- Step 4: Mock stats (these can be computed properly later) ---
    total_students = int(df["Student ID"].nunique())
    stats = {
        "3_per_day": int(total_students * 0.05),
        "4_per_day": int(total_students * 0.02),
        "3_consecutive": int(total_students * 0.03),
        "4_consecutive": int(total_students * 0.01)
    }

    # --- Step 5: Return all outputs ---
    return {
        "course_schedule": schedule_df,
        "students_per_slot": students_per_slot,
        "stats": stats
    }
