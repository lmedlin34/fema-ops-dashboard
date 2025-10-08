🧭 README — FEMA Disaster Declarations Dashboard

Overview

The FEMA Disaster Declarations Dashboard is a real-time operational analytics app that visualizes disaster declarations across the United States using FEMA’s open data API.
It provides emergency operations analysts, program managers, and decision-makers with immediate situational awareness of disaster activity — by state, type, and time — through a clean and responsive interface.

Built entirely in Python + Streamlit, this app demonstrates modern data-product design principles:  
	•	API-driven data retrieval  
	•	Geospatial visualization via pydeck heatmaps  
	•	Real-time KPIs and trends  
	•	Interactive filtering and export  

⸻

🎯 Purpose

To showcase how operational open data can be transformed into an interactive, insight-ready tool that supports quick assessments of disaster load and recency.

In a production environment, this dashboard could power:  
	•	Workforce surge planning for FEMA regions  
	•	Disaster trend analysis for annual reports  
	•	Performance metrics for federal and state emergency management agencies  

⸻

⚙️ Features

Feature	Description  
🕓 Live Data Fetch	Connects to FEMA’s DisasterDeclarationsSummaries API and refreshes on demand.  
🧩 KPI Overview	Displays total declarations, new in last 30 days, and average days since last declaration.  
🗺️ Interactive Heatmap	County-level aggregation using 5-digit FIPS centroids from the U.S. Census Gazetteer.  
📈 Trend Visualization	Monthly declaration trends and incident-type composition via Plotly.  
🧮 Smart Filters	State, fiscal year, incident type, and declaration type — all applied dynamically.  
💾 Export & Transparency	Download filtered data as CSV and view full methodology notes inside the app.  


⸻

🧠 Methods & Data  
	•	Source: FEMA OpenFEMA API → Disaster Declarations Summaries  
	•	Geography: U.S. Census Bureau Gazetteer county centroids (2021–2023 editions)  
	•	FIPS Join: FEMA fipsStateCode + fipsCountyCode → fipsCountyCodeFull (5-digit)  
	•	KPIs:  
	•	Total Declarations: all records in time window  
	•	New in 30 Days: declarations in last 30 days, with delta vs. prior 30-day window  
	•	Average Days Since: mean elapsed time since most recent declaration  
	•	Tech stack: Streamlit, pandas, plotly, pydeck, requests  

⸻

🧱 Architecture

app.py                # Main Streamlit app  
requirements.txt      # Dependencies  
data/fema_demo.csv    # Optional fallback dataset  
.streamlit/config.toml# Theme + layout settings  


⸻

🚀 Getting Started

Prerequisites  
	•	Python 3.11+  
	•	pip (or conda)  

Installation

git clone https://github.com/yourusername/fema-ops-dashboard.git  
cd fema-ops-dashboard  
pip install -r requirements.txt  

Run the App

streamlit run app.py

Optional:

If your network blocks Census Gazetteer URLs, upload your own CSV in-app under
“Geo lookup (optional)” → Upload FIPS→lat/lon CSV
(columns: fipsCountyCode, latitude, longitude).

⸻

📸 Screenshots

(Add 2–3 screenshots here — e.g., Overview KPIs, Heatmap zoomed to NC, Trends by incident type.)

⸻

🧩 Future Enhancements  
	•	Disaster forecast module using ARIMA on historical time series  
	•	Slack/Teams alert integration for new declarations  
	•	Dynamic clustering by severity (PA/IH/HM program declared)  
	•	Federal–state response correlation dashboard  

⸻

👨‍💻 Author

Luke Medlin
www.lukemedlin.com
Data Scientist 
