import streamlit as st
from phishing.ui import render_phishing_tab

# Page Configuration
st.set_page_config(page_title="Dual-Threat Triage System", layout="wide")

# Main Title
st.title("🛡️ Dual-Threat Triage System")
st.markdown("An Explainable ML System for CVE Prioritization and Zero-Day Phishing Detection.")

# Create Tabs (Placeholder for Aiman's CVE tab later)
tab_phishing, tab_cve = st.tabs([" Phishing Detector", " CVE Triage (Coming Soon)"])

# Render your half in the first tab
with tab_phishing:
    render_phishing_tab()

# Placeholder for the second tab
with tab_cve:
    st.info("CVE Triage module is currently under development by the team.")