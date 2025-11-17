# FAST Peshawar Sessional 1 Scheduling Report

## Scheduler Verification
- The scheduler now colours the exam conflict graph with a backtracking graph-colouring algorithm that enforces the 500-student capacity bound, minimises daily overloads, and discourages consecutive slots while it searches.
- Running `schedule_exams` on the provided `Student Data.xlsx` produces a **clash-free** timetable: no student is assigned two exams in the same slot.
- All 18 slots stay within capacity and the fairness penalties drop dramatically (see metrics below). The Streamlit UI only offers download buttons; no files are written automatically to the workspace.

## Graph-Theoretic Formulation
- **Conflict Graph:** Each `(Course Code, Subject Name)` pair is a vertex. Two vertices share an edge iff at least one student is enrolled in both courses. A proper colouring with ≤18 colours corresponds to a clash-free schedule. Vertex weights equal cohort sizes, so colour classes must respect the 500-student capacity constraint.
- **Backtracking Colouring:** Courses are ordered by a hybrid of graph degree and cohort size. For each course, feasible slots are explored in ascending penalty order. The penalty function counts (i) how many students would exceed two papers in a day and (ii) how many would gain back-to-back or three-in-a-row exams if the slot were chosen. This mirrors minimising high-degree vertices in the student-day projection graph.
- **Search Guarantees:** Because the search backtracks, it can reshuffle earlier decisions whenever a future course cannot be placed, ensuring feasibility provided the input admits a valid colouring. The heuristic penalties steer the search toward balanced day loads without compromising correctness.

## Empirical Metrics (Final Schedule)
### a. Students per Slot
| Day | Slot | Students |
| --- | --- | --- |
| 1 | 1 | 399 |
| 1 | 2 | 152 |
| 1 | 3 | 483 |
| 1 | 4 | 200 |
| 1 | 5 | 468 |
| 1 | 6 | 200 |
| 2 | 1 | 412 |
| 2 | 2 | 249 |
| 2 | 3 | 203 |
| 2 | 4 | 149 |
| 2 | 5 | 412 |
| 2 | 6 | 433 |
| 3 | 1 | 488 |
| 3 | 2 | 96 |
| 3 | 3 | 322 |
| 3 | 4 | 246 |
| 3 | 5 | 224 |
| 3 | 6 | 492 |

### b. Students with ≥3 Papers on One Day
| Papers in a Single Day | Students Affected |
| --- | --- |
| 3 papers | 225 |
| 4 papers | 0 |
| 5 papers | 0 |
| 6 papers | 0 |

Counts are aggregated per student per day; zero indicates no student experiences four or more papers on the same day.

### c. Students with ≥3 Consecutive Papers (Same Day)
| Length of Consecutive Run | Students Affected |
| --- | --- |
| 3 consecutive slots | 1 |
| 4 consecutive slots | 0 |
| 5 consecutive slots | 0 |
| 6 consecutive slots | 0 |

Only a single student encounters a run of three consecutive slots, and no longer runs occur.

## Recommendations
- Fine-tune penalty weights or introduce adaptive weights per student cohort if further fairness improvements are required.
- Add optional reports (via the UI downloads) that highlight the single remaining consecutive-run case so administrators can decide whether to manually adjust.
- The current implementation is modular: alternative heuristics (e.g. ILP solver or metaheuristics) can be plugged into the `_candidate_slots` exploration step if future datasets prove more demanding.
