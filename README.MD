# XCAPE

## Automated History Matching Platform for Reservoir Performance Analysis and Forecasting

XCAPE is a final-year project focused on the development of an accessible, educational, and deployable platform for reservoir history matching and forecasting. The project integrates the open-source reservoir simulator OPM Flow with automated calibration techniques, particularly the Ensemble Kalman Filter (EnKF), alongside a manual baseline matching workflow. The system is designed to improve the understanding and application of reservoir simulation by combining technical performance analysis with a user-friendly web interface.

This platform brings together Python, FastAPI, React, and PostgreSQL to support simulation execution, automated parameter estimation, result visualization, and deployment in a teaching-oriented environment.

---

## Project Information

- Project Topic: Development of a Python-Based Automated History Matching Tool for Reservoir Performance Analysis and Forecasting
- Student: Osimi Godsgift Gbubemi
- Registration Number: 20211276493
- Supervisor: Dr. Chikwe
- Department: Petroleum Engineering
- Project Type: Final Year Project (Individual)
- Core Stack: OPM Flow, Python, FastAPI, EnKF, React (Vite), PostgreSQL, Render

---

## Abstract

This project develops an accessible, educational, and deployable platform that automates reservoir history matching to improve reservoir performance analysis and forecasting. The tool integrates the open-source reservoir simulator OPM Flow with automated calibration algorithms, notably the Ensemble Kalman Filter (EnKF), and a baseline manual matching option. The system produces predictions that align reservoir models with real production data, quantifies prediction uncertainty, and provides a web interface for interactive visualization and teaching.

---

## Problem and Importance

Translating field measurements such as pressures and production rates into reliable reservoir forecasts is often a manual, time-consuming, and technically demanding process. In many cases, these workflows are inaccessible to students due to licensing restrictions, model complexity, and limited practical resources.

Inaccurate models can lead to misleading forecasts and poor field decisions. This project aims to build a reproducible, low-cost, and educational platform that automates reservoir matching against observed data and supports forecasting in a more transparent and accessible manner.

---

## Aim

To build and demonstrate a Python-based automated history matching tool that aligns reservoir model predictions with real production data for improved performance analysis and forecasting.

---

## Objectives

1. Implement a baseline manual matching interface to illustrate traditional parameter tuning methods and their limitations.
2. Implement an Ensemble Kalman Filter (EnKF) workflow to automate parameter estimation and provide uncertainty quantification.
3. Integrate OPM Flow with a FastAPI backend to execute forward simulations against observed or synthetic production data.
4. Generate forecast runs that compare prior and posterior model predictions, emphasizing improvements in predictive accuracy.
5. Develop a simple React frontend for interactive visualization aimed at educational use and open-access learning.
6. Deploy the demonstration platform on Render to enable student and lecturer access.

---

## Methodology and Tools

### Data
- Synthetic datasets and benchmark cases
- Publicly available reservoir examples
- Optional small real-field or anonymized datasets for demonstration

### Simulator
- OPM Flow used for forward model simulation

### Algorithms
- Baseline manual matching
- Ensemble Kalman Filter (EnKF) for automated calibration and uncertainty estimation

### Backend/API
- FastAPI for orchestrating jobs, managing ensembles, and serving results to the frontend

### Frontend
- React (Vite)
- Recharts or Plotly for interactive visualization
- Material UI for a professional and consistent interface

### Database and Deployment
- PostgreSQL for metadata and result storage
- Render for cloud deployment and accessibility

---

## Expected Outcomes and Deliverables

- A web-accessible demo for baseline and EnKF-based history matching
- Forecast outputs comparing prior and posterior model predictions
- Visualization of observed versus simulated data, ensemble spread, and posterior distributions
- Source code repository with reproducible instructions and deployment setup
- Student guide for practical use and demonstration
- Project report and recorded demonstration for academic presentation support
- Deployed platform for student and lecturer exploration

---

## High-Level Timeline

- Week 1: Dataset review and simulator validation
- Weeks 2–3: Backend development and baseline matching
- Week 4: EnKF implementation and testing
- Week 5: Frontend integration and visualization
- Week 6: Deployment, documentation, and rehearsal

---

## Frontend Requirements

The platform should include a modern, professional web experience with the following pages:

### Home Page
A polished landing page presenting the platform concept, reservoir simulation overview, and project purpose with concise educational content and visual illustrations.

### Concepts Page
A paginated educational section covering major reservoir simulation and history matching concepts with expandable detailed views.

### User Guide Page
A practical guide for using the platform to perform simulations and interpret results.

### XCAPE Simulator Page
The main interactive dashboard where users can run analysis workflows, visualize results, and compare model performance.

### About Page
A detailed overview of the project’s motivation, problem statement, and objectives.

### Profiles and Contributors Page
A contributor section showing academic and professional details, including:
- image
- qualification
- bio
- achievements
- role
- contact information
- address

### Contact Page
A dedicated section for university, supervisor, and project-related contact information.

---

## Design Direction

The user interface should be modern, field-oriented, and highly professional, with strong emphasis on clarity and usability.

### Color Palette
- Primary: #0F4C81
- Secondary: #1F7A8C
- Accent: #F4B400
- Background: #F7F9FC
- Card Surface: #FFFFFF
- Text: #1F2937
- Success: #22C55E
- Warning: #F59E0B
- Error: #EF4444

### Frontend Standards
- Use Material UI
- Proper use of React Hooks and Redux
- Use modern frontend tooling and optimization practices
- Include dark mode and light mode toggle
- Use strong branding for XCAPE on the home page

---

## Project Files and Paths

### Frontend Folder
`C:\Users\USER\Desktop\XCAPE\frontend`

### Backend Folder
`C:\Users\USER\Desktop\XCAPE\backend\XCAPE`

### Logos
- White Background: `C:\Users\USER\Desktop\XCAPE\frontend\Xcape.png`
- Black Background: `C:\Users\USER\Desktop\XCAPE\frontend\xcapeblackBG.png`

### Images Folder
`C:\Users\USER\Desktop\XCAPE\frontend\images`

---

## Technical Notes

- XCAPE is the name of the project platform and should be clearly visible in the home page branding.
- The frontend should use modern, professional styling and a strong visual identity.
- The project is intended as both a functional tool and an educational platform for reservoir engineering learners.

---

## Student and Supervisor Details

### Student
Osimi Godsgift Gbubemi  
Registration No.: 20211276493

### Supervisor
Dr. Chikwe

### Department
Petroleum Engineering

---

## Conclusion

This project demonstrates the integration of reservoir simulation, data assimilation, and modern web technologies to create an educational and practical platform for automated history matching. By combining open-source tools with a clear user interface, XCAPE aims to support learning, improve decision-making, and provide a scalable foundation for future extension and research.

---

Prepared by: Osimi Godsgift Gbubemi  
Registration No.: 20211276493  
Supervisor: Dr. Chikwe
