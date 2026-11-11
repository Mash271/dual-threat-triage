import streamlit as st
import plotly.graph_objects as go
from .phishing_model import predict

# --- Demo URLs for easy testing during presentations ---
DEMO_URLS = {
    "Select a demo URL...": "",
    "Safe: Google Search": "https://www.google.com/search?q=cats",
    "Safe: GitHub Repository": "https://github.com/yourname/dual-threat-triage",
    "Phishing: IP Address Login": "http://192.168.1.1/login-secure/verify.php?user=admin",
    "Phishing: Suspicious Subdomain": "http://secure-paypal-verify.xyz/update-account",
    "Phishing: Long & Complex": "http://a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z.com/login"
}

def render_phishing_tab():
    st.header("🎣 Phishing URL Detector")
    st.markdown("Paste a URL below to analyze its structural risk and detect zero-day phishing attempts.")
    
    # 1. Input Section
    col1, col2 = st.columns([3, 1])
    with col1:
        user_url = st.text_input("Enter URL to scan:", placeholder="https://example.com")
    with col2:
        demo_choice = st.selectbox("Or try a demo:", list(DEMO_URLS.keys()))
    
    # Sync the text input with the dropdown
    if demo_choice and DEMO_URLS[demo_choice] != "":
        user_url = DEMO_URLS[demo_choice]

    # 2. Analysis Button
    if st.button("Analyze URL", type="primary", use_container_width=True):
        if not user_url:
            st.error("Please enter a URL first.")
            return
            
        with st.spinner("Analyzing URL structure and calculating risk..."):
            # Call the contract-compliant prediction engine
            result = predict(user_url)
            
        # 3. Display Results
        score = result["risk_score"]
        verdict = result["verdict"]
        reasons = result["reasons"]
        
        # --- Top Row: Gauge and Verdict ---
        col_gauge, col_verdict = st.columns([2, 1])
        
        with col_gauge:
            # Create a beautiful Plotly Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Risk Score (0-100)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},   # Safe zone
                        {'range': [30, 70], 'color': "yellow"},      # Warning zone
                        {'range': [70, 100], 'color': "red"}         # Danger zone
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_verdict:
            st.subheader("Verdict")
            if verdict == "BLOCK":
                st.error(f"🚨 {verdict}")
                st.caption("High confidence phishing detected.")
            else:
                st.success(f"✅ {verdict}")
                st.caption("URL appears structurally safe.")

        # --- Bottom Row: Explainability (SHAP Reasons) ---
        st.markdown("---")
        st.subheader("🔍 Why did the model flag this?")
        st.markdown("The following structural features most influenced the prediction:")
        
        for i, reason in enumerate(reasons, 1):
            st.write(f"**{i}.** {reason}")
