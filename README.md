# FAST Peshawar Exam Scheduler

This project provides an intelligent exam scheduling system that generates clash-free exam timetables. It uses graph-coloring techniques to prevent overlapping exams, respect capacity limits, and reduce student workload.

Live Preview: [(p229021.streamlit.app)](https://p229021.streamlit.app/)

## Features
- Clash-free scheduling based on student enrollments  
- Maximum 500 students per slot  
- Minimizes daily exam overload and consecutive exams  
- Conflict graph and timetable heatmap visualizations  
- Downloadable Excel and Markdown reports  

## Requirements
- Python 3.8+
- pandas  
- streamlit  
- networkx  
- plotly  
- xlsxwriter / openpyxl  
- kaleido  

## Installation
```bash
git clone <repository-url>
```
```bash
cd GT_Assignment_02
```
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```


## Uploading Data
Upload an Excel file containing the following columns:

- Student ID  
- Course Code  
- Section  
- Subject Name  

The system will automatically generate the final exam schedule, statistics, heatmaps, and downloadable reports.


## Scheduling Configuration

- 6 slots per day  
- 3 days total  
- 18 slots overall  
- Maximum 500 students per slot  


## Algorithm Overview

The scheduler constructs a conflict graph, orders courses by degree and cohort size, assigns slots using a penalty-based scoring method, and applies backtracking to ensure a valid, clash-free schedule.


## Output Files

- **Exam_Schedule.xlsx**  
- **Students_per_Slot.xlsx**  
- **Student_Schedule.xlsx**  
- **DESCRIPTION.md** (detailed statistics)


## License

This project is part of an academic assignment.

## Author

Developed by **[Uzair Ahmad](https://uzairkbrr.netlify.app/)**
