import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Cinema Fare AI | Dynamic Box-Office Engine",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Cinema Theme)
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f293d 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .price-display {
        font-size: 3rem;
        font-weight: 800;
        color: #f59e0b;
        margin: 10px 0;
    }
    .concession-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .tag-child { background-color: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
    .tag-youth { background-color: #3b2a59; color: #c084fc; border: 1px solid #9333ea; }
    .tag-senior { background-color: #3f2e22; color: #f97316; border: 1px solid #ea580c; }
    .tag-standard { background-color: #1f2937; color: #9ca3af; border: 1px solid #4b5563; }
    </style>
""", unsafe_allow_html=True)

# Load serialized model pipeline
@st.cache_resource
def load_pricing_pipeline():
    return joblib.load("ticket_pricing_model_compressed.pkl")

try:
    model_pipeline = load_pricing_pipeline()
except FileNotFoundError:
    st.error("⚠️ Model file `ticket_pricing_model_compressed.pkl` not found in working directory.")
    st.stop()

# Sidebar: Booking Parameters
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🎬 Booking Preferences")
    
    age = st.slider("Viewer Age", min_value=3, max_value=85, value=24, step=1)
    
    city_tier = st.selectbox(
        "Location / Urban Classification",
        ["Tier 1 (Metro)", "Tier 2", "Tier 3"],
        index=0,
        help="Metro (Mumbai, Delhi, Bengaluru, etc.) vs. Regional Cities"
    )
    
    screen_type = st.selectbox(
        "Auditorium Format",
        ["Standard 2D", "3D", "IMAX", "4DX", "Gold/Recliner"],
        index=0
    )
    
    day_type = st.radio("Showday", ["Weekday", "Weekend"], horizontal=True)
    
    show_time = st.select_slider(
        "Show Timing Window",
        options=["Morning", "Matinee/Afternoon", "Prime Evening", "Night"],
        value="Prime Evening"
    )

# Demographic Analysis Logic
if age < 12:
    badge_html = '<span class="concession-pill tag-child">🧒 Child Concession Applied (~35% discount)</span>'
elif 18 <= age <= 24:
    badge_html = '<span class="concession-pill tag-youth">🎓 Student / Youth Saver (~10% discount)</span>'
elif age >= 60:
    badge_html = '<span class="concession-pill tag-senior">👴 Senior Citizen Special (~30% discount)</span>'
else:
    badge_html = '<span class="concession-pill tag-standard">🎟️ Standard Adult Fare</span>'

# Prediction Inference
input_df = pd.DataFrame([{
    'Age': age,
    'City_Tier': city_tier,
    'Screen_Type': screen_type,
    'Day_Type': day_type,
    'Show_Time': show_time
}])

predicted_base = float(model_pipeline.predict(input_df)[0])

# Indian Statutory Cinema GST Calculation (12% if <= ₹100, 18% if > ₹100)
gst_rate = 0.12 if predicted_base <= 100 else 0.18
gst_amount = predicted_base * gst_rate
total_price = round(predicted_base + gst_amount)

# Layout Presentation
st.title("🇮🇳 India Cinema Ticket Price Estimator")
st.caption("Machine Learning inference engine for demographic and dynamic theater pricing across Indian screens.")
st.divider()

col_main, col_breakdown = st.columns([1.2, 1])

with col_main:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color: #9ca3af; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Calculated Ticket Fare</div>
            <div class="price-display">₹{total_price:,.0f}</div>
            {badge_html}
            <div style="margin-top: 15px; color: #6b7280; font-size: 0.85rem;">Inclusive of applicable Central & State Entertainment GST</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("")
    with st.expander("🔍 View Raw Pipeline Input Features"):
        st.dataframe(input_df, hide_index=True, use_container_width=True)

with col_breakdown:
    st.subheader("Invoice Breakdown")
    
    fare_df = pd.DataFrame({
        "Cost Component": [
            "ML Base Fare Prediction",
            f"Statutory Entertainment GST ({int(gst_rate * 100)}%)",
            "Total Payable"
        ],
        "Amount (INR)": [
            f"₹{predicted_base:.2f}",
            f"₹{gst_amount:.2f}",
            f"₹{total_price:.2f}"
        ]
    })
    
    st.table(fare_df)
    
    st.info(
        f"**Dynamic Pricing Factors Identified:**\n"
        f"* **Market Tier:** {city_tier}\n"
        f"* **Experience Format:** {screen_type}\n"
        f"* **Schedule Surge:** {day_type} ({show_time})"
    )
