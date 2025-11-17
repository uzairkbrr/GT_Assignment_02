"""
Exam scheduler utilities.

Provides a simple clash-free scheduler function used by `app.py`.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# --- Scheduling configuration ---
SLOTS_PER_DAY = 6
TOTAL_DAYS = 3
MAX_STUDENTS_PER_SLOT = 500
TOTAL_SLOTS = SLOTS_PER_DAY * TOTAL_DAYS
SLOT_LABELS = [f"Day {day + 1} - Slot {slot + 1}" for day in range(TOTAL_DAYS) for slot in range(SLOTS_PER_DAY)]


@dataclass(frozen=True)
class CourseKey:
    """
    Unique identifier for a course in the schedule.
    """
    course_code: str
    subject_name: str


def _normalize_column_name(name: str) -> str:
    """
    Lower-case, strip punctuation, and collapse whitespace to make column matching tolerant.
    """
    cleaned = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in str(name))
    return ' '.join(cleaned.split())


def _match_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns of df to canonical names required for scheduling.
    """
    required_cols = ["Student ID", "Course Code", "Section", "Subject Name"]
    alias_map = {
        "Student ID": ["roll no", "rollno", "reg no", "regno", "student no", "studentid", "id"],
        "Course Code": ["code", "coursecode", "course code", "course_code"],
        "Section": ["section", "sec"],
        "Subject Name": ["course", "subject", "subject name", "course name", "course_name"]
    }

    normalized_columns = {_normalize_column_name(col): col for col in df.columns}

    alias_lookup = {}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            alias_lookup[_normalize_column_name(alias)] = canonical

    matched_columns: Dict[str, str] = {}
    
    import difflib

    for required in required_cols:
        normalized_required = _normalize_column_name(required)

        if normalized_required in normalized_columns:
            matched_columns[required] = normalized_columns[normalized_required]
            continue

        matched_alias = next(
            (original for norm, original in normalized_columns.items() if alias_lookup.get(norm) == required), None
        )
        if matched_alias:
            matched_columns[required] = matched_alias
            continue

        candidates = difflib.get_close_matches(
            normalized_required, list(normalized_columns.keys()), n=1, cutoff=0.75
        )
        if candidates:
            matched_columns[required] = normalized_columns[candidates[0]]

    missing = [col for col in required_cols if col not in matched_columns]
    if missing:
        available = ', '.join(df.columns)
        raise ValueError(f"Missing required column(s): {', '.join(missing)}. Available columns: {available}")

    return df.rename(columns={original: canonical for canonical, original in matched_columns.items()})


def _build_course_roster(df: pd.DataFrame) -> Tuple[Dict[CourseKey, Set[Any]], Dict[Any, List[CourseKey]]]:
    """
    Build mappings from courses to enrolled students and vice versa.
    """
    course_to_students: Dict[CourseKey, Set[Any]] = defaultdict(set)
    student_to_courses: Dict[Any, List[CourseKey]] = defaultdict(list)

    for _, row in df.iterrows():
        key = CourseKey(course_code=str(row["Course Code"]).strip(), subject_name=str(row["Subject Name"]).strip())
        student_id = row["Student ID"]
        course_to_students[key].add(student_id)

    for course, students in course_to_students.items():
        for student in students:
            student_to_courses[student].append(course)

    return course_to_students, student_to_courses


def _build_conflict_graph(student_to_courses: Dict[Any, List[CourseKey]]) -> Dict[CourseKey, Set[CourseKey]]:
    """
    Construct an undirected conflict graph where edges connect courses sharing a student.
    """
    conflicts: Dict[CourseKey, Set[CourseKey]] = defaultdict(set)
    for courses in student_to_courses.values():
        for course_a, course_b in combinations(courses, 2):
            conflicts[course_a].add(course_b)
            conflicts[course_b].add(course_a)
    return conflicts


def schedule_exams(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a clash-free exam timetable using graph colouring with capacity constraints.

    Inputs:
    - df: DataFrame with columns ["Student ID", "Course Code", "Section", "Subject Name"]

    Outputs:
    - course_schedule: DataFrame mapping each course to a slot and student count
    - students_per_slot: DataFrame with total students per slot
    - stats: dictionary containing daily load and consecutive exam statistics
    """

    df = _match_columns(df)
    df = df.dropna(subset=["Student ID", "Course Code", "Subject Name"])

    course_to_students, student_to_courses = _build_course_roster(df)
    if not course_to_students:
        raise ValueError("No courses found in the provided data.")

    conflicts = _build_conflict_graph(student_to_courses)
    course_sizes = {course: len(students) for course, students in course_to_students.items()}

    degree: Dict[CourseKey, int] = {course: len(conflicts.get(course, set())) for course in course_to_students}

    courses_order: List[CourseKey] = sorted(
        course_to_students.keys(),
        key=lambda course: (degree.get(course, 0), course_sizes[course]),
        reverse=True,
    )

    assigned_slots: Dict[CourseKey, int] = {}
    slot_loads: List[int] = [0 for _ in range(TOTAL_SLOTS)]

    student_assignments: Dict[Any, List[int]] = defaultdict(list)
    student_day_counts: Dict[Any, List[int]] = defaultdict(lambda: [0 for _ in range(TOTAL_DAYS)])
    student_day_slots: Dict[Any, List[Set[int]]] = defaultdict(lambda: [set() for _ in range(TOTAL_DAYS)])

    def conflicts_with_neighbour(course: CourseKey, slot_idx: int) -> bool:
        return any(assigned_slots.get(neighbour) == slot_idx for neighbour in conflicts.get(course, set()))

    def slot_penalty(course: CourseKey, slot_idx: int) -> Tuple[int, int, int, int]:
        course_size = course_sizes[course]
        day_idx = slot_idx // SLOTS_PER_DAY
        slot_in_day = slot_idx % SLOTS_PER_DAY

        overload_penalty = 0
        consecutive_penalty = 0

        for student in course_to_students[course]:
            day_count = student_day_counts[student][day_idx]
            if day_count >= 2:
                overload_penalty += 5
            elif day_count == 1:
                overload_penalty += 1

            day_slots = student_day_slots[student][day_idx]
            if day_slots:
                left1 = slot_in_day - 1
                left2 = slot_in_day - 2
                right1 = slot_in_day + 1
                right2 = slot_in_day + 2
                if ({left1, left2} <= day_slots) or ({right1, right2} <= day_slots) or (left1 in day_slots and right1 in day_slots):
                    consecutive_penalty += 5
                elif left1 in day_slots or right1 in day_slots:
                    consecutive_penalty += 1

        residual_capacity = MAX_STUDENTS_PER_SLOT - (slot_loads[slot_idx] + course_size)
        return (
            overload_penalty + consecutive_penalty,
            slot_loads[slot_idx],
            -residual_capacity,
            slot_idx,
        )

    def assign(course: CourseKey, slot_idx: int) -> None:
        assigned_slots[course] = slot_idx
        slot_loads[slot_idx] += course_sizes[course]
        day_idx = slot_idx // SLOTS_PER_DAY
        slot_in_day = slot_idx % SLOTS_PER_DAY
        for student in course_to_students[course]:
            student_assignments[student].append(slot_idx)
            student_day_counts[student][day_idx] += 1
            student_day_slots[student][day_idx].add(slot_in_day)

    def unassign(course: CourseKey, slot_idx: int) -> None:
        del assigned_slots[course]
        slot_loads[slot_idx] -= course_sizes[course]
        day_idx = slot_idx // SLOTS_PER_DAY
        slot_in_day = slot_idx % SLOTS_PER_DAY
        for student in course_to_students[course]:
            student_assignments[student].remove(slot_idx)
            student_day_counts[student][day_idx] -= 1
            student_day_slots[student][day_idx].discard(slot_in_day)

    def candidate_slots(course: CourseKey) -> List[Tuple[Tuple[int, int, int, int], int]]:
        candidates: List[Tuple[Tuple[int, int, int, int], int]] = []
        for slot_idx in range(TOTAL_SLOTS):
            if slot_loads[slot_idx] + course_sizes[course] > MAX_STUDENTS_PER_SLOT:
                continue
            if conflicts_with_neighbour(course, slot_idx):
                continue
            candidates.append((slot_penalty(course, slot_idx), slot_idx))
        candidates.sort()
        return candidates

    def backtrack(position: int) -> bool:
        if position == len(courses_order):
            return True

        course = courses_order[position]
        slots = candidate_slots(course)
        if not slots:
            return False

        for _, slot_idx in slots:
            assign(course, slot_idx)
            if backtrack(position + 1):
                return True
            unassign(course, slot_idx)

        return False

    if not backtrack(0):
        raise ValueError("Unable to compute a clash-free schedule within capacity constraints.")

    schedule_rows: List[Dict[str, Any]] = []
    for course, slot_idx in assigned_slots.items():
        schedule_rows.append(
            {
                "Course Code": course.course_code,
                "Subject Name": course.subject_name,
                "Slot": SLOT_LABELS[slot_idx],
                "Students": course_sizes[course],
            }
        )

    schedule_df = pd.DataFrame(schedule_rows).sort_values("Slot").reset_index(drop=True)

    students_per_slot = pd.DataFrame(
        [{"Slot": SLOT_LABELS[idx], "Students": load} for idx, load in enumerate(slot_loads)]
    ).sort_values("Slot")

    daily_load_counter: Counter = Counter()
    consecutive_counter: Counter = Counter()

    for student, slots in student_assignments.items():
        slots_by_day: Dict[int, List[int]] = defaultdict(list)
        for slot_idx in slots:
            day = slot_idx // SLOTS_PER_DAY
            day_slot = slot_idx % SLOTS_PER_DAY
            slots_by_day[day].append(day_slot)

        for day_slots in slots_by_day.values():
            day_slots.sort()
            load = len(day_slots)
            if load >= 3:
                daily_load_counter[min(load, 6)] += 1

            run_length = 1
            longest_run = 1
            for a, b in zip(day_slots, day_slots[1:]):
                if b == a + 1:
                    run_length += 1
                    longest_run = max(longest_run, run_length)
                else:
                    run_length = 1

            for length in range(3, min(longest_run, 6) + 1):
                consecutive_counter[length] += 1

    stats = {
        "3_per_day": daily_load_counter.get(3, 0),
        "4_per_day": daily_load_counter.get(4, 0),
        "5_per_day": daily_load_counter.get(5, 0),
        "6_per_day": daily_load_counter.get(6, 0),
        "3_consecutive": consecutive_counter.get(3, 0),
        "4_consecutive": consecutive_counter.get(4, 0),
        "5_consecutive": consecutive_counter.get(5, 0),
        "6_consecutive": consecutive_counter.get(6, 0),
        "daily_load_counts": dict(daily_load_counter),
        "consecutive_counts": dict(consecutive_counter),
    }

    return {
        "course_schedule": schedule_df,
        "students_per_slot": students_per_slot,
        "stats": stats,
    }
