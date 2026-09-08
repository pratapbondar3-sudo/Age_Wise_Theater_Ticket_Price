import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Cinema Fare AI | All-India Dynamic Box-Office Engine",
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

# List of comprehensive Indian cities categorized by cinema pricing dynamics
INDIAN_CITIES = [
   ADDITIONAL_CITIES = [
    # South India
    "Vijayawada", "Guntur", "Tirupati", "Warangal", "Mangaluru", 
    "Hubballi-Dharwad", "Kozhikode", "Thrissur", "Kollam", "Kannur", 
    "Tiruchirappalli", "Salem", "Tirunelveli", "Vellore",
    
    # North & Central
    "Kanpur", "Prayagraj", "Meerut", "Bareilly", "Gorakhpur", 
    "Ludhiana", "Jalandhar", "Jodhpur", "Udaipur", "Kota", 
    "Gwalior", "Jabalpur", "Raipur", "Jammu", "Srinagar",
    
    # West
    "Nashik", "Chhatrapati Sambhaji Nagar", "Kolhapur", "Solapur", 
    "Rajkot", "Bhavnagar", "Panaji","Dharashiv","Latur","Beed",
    
    # East & North-East
    "Siliguri", "Asansol", "Durgapur", "Jamshedpur", "Dhanbad", 
    "Cuttack", "Gaya", "Muzaffarpur", "Shillong"
]

THEATER_CHAINS = [
    "PVR INOX Multiplex", "Cinepolis", "Miraj Cinemas", "Carnival Cinemas", 
    "MovieMax", "Wave Cinemas", "SRS Cinemas", "Single Screen Heritage Theater"
]

ALL_MONTHS = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# Cache the trained pipeline directly inside the runtime environment
@st.cache_resource(show_spinner="Training All-India Pricing Engine...")
def get_trained_pipeline():
    np.random.seed(42)
    n_samples = 8000

    ages = np.random.randint(3, 80, size=n_samples)
    cities = np.random.choice(INDIAN_CITIES, size=n_samples)
    theaters = np.random.choice(THEATER_CHAINS, size=n_samples)
    screen_types = np.random.choice(['Standard 2D', '3D', 'IMAX', '4DX', 'Gold/Recliner'], size=n_samples, p=[0.40, 0.25, 0.15, 0.08, 0.12])
    months = np.random.choice(ALL_MONTHS, size=n_samples)
    days = np.random.choice(DAYS_OF_WEEK, size=n_samples)
    show_times = np.random.choice(['Morning', 'Matinee/Afternoon', 'Prime Evening', 'Night'], size=n_samples, p=[0.2, 0.25, 0.4, 0.15])

    tier_1_cities = {"Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"}
    tier_2_cities = {"Chandigarh", "Jaipur", "Lucknow", "Kochi", "Indore", "Bhopal", "Nagpur", "Surat", "Visakhapatnam", "Coimbatore", "Vadodara"}
    
    # Holiday / festive peak release months (Diwali, Eid, Christmas, Summer Holidays)
    peak_months = {"May", "June", "October", "November", "December"}

    base_prices = []
    for age_val, city, theater, screen, month, day, show in zip(ages, cities, theaters, screen_types, months, days, show_times):
        price = 170.0
        
        # Location Tier Adjustments
        if city in tier_1_cities:
            price += 90
        elif city in tier_2_cities:
            price += 40
        else:
            price += 10
            
        # Cinema Brand Category
        if "PVR INOX" in theater or "Cinepolis" in theater:
            price += 50
        elif "Single Screen" in theater:
            price -= 50
            
        # Format Additions
        if screen == '3D': price += 60
        elif screen == 'IMAX': price += 220
        elif screen == '4DX': price += 280
        elif screen == 'Gold/Recliner': price += 350
        
        # Day surges
        if day in ['Saturday', 'Sunday']:
            price += 70
        elif day == 'Friday':
            price += 30
        elif day in ['Tuesday', 'Wednesday']:
            price -= 20  # Mid-week discount runs
            
        # Show Time Impact
        if show == 'Prime Evening': price += 40
        elif show == 'Morning': price -= 40
        
        # Month / Seasonal Release Surge
        if month in peak_months:
            price += 30

        # Concessions by Age Bracket
        if age_val < 12:
            price *= 0.65
        elif age_val >= 60:
            price *= 0.70
        elif 18 <= age_val <= 24:
            price *= 0.90
            
        price += np.random.normal(0, 15)
        base_prices.append(max(70.0, round(price, 2)))

    df = pd.DataFrame({
        'Age': ages,
        'City': cities,
        'Theater': theaters,
        'Screen_Type': screen_types,
        'Month': months,
        'Day': days,
        'Show_Time': show_times,
        'Ticket_Price_INR': base_prices
    })

    X = df.drop('Ticket_Price_INR', axis=1)
    y = df['Ticket_Price_INR']

    categorical_cols = ['City', 'Theater', 'Screen_Type', 'Month', 'Day', 'Show_Time']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# Train / retrieve cached model in current runtime
model_pipeline = get_trained_pipeline()

# Sidebar Inputs
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🎬 All-India Cinema Booking")
    
    city = st.selectbox("Select City", sorted(INDIAN_CITIES), index=0)
    theater = st.selectbox("Theater / Chain", THEATER_CHAINS, index=0)
    
    booking_date = st.date_input(
        "Booking Date",
        value=datetime.date.today(),
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=365)
    )
    
    # Automatically compute Day Name and Month Name from Selected Date
    day_name = booking_date.strftime("%A")
    month_name = booking_date.strftime("%B")
    
    st.caption(f"📅 **Selected Day:** {day_name} | **Month:** {month_name}")

    age = st.slider("Viewer Age", min_value=3, max_value=85, value=24, step=1)
    
    screen_type = st.selectbox(
        "Auditorium Format",
        ["Standard 2D", "3D", "IMAX", "4DX", "Gold/Recliner"],
        index=0
    )
    
    show_time = st.select_slider(
        "Show Timing Window",
        options=["Morning", "Matinee/Afternoon", "Prime Evening", "Night"],
        value="Prime Evening"
    )

# Badge Logic
if age < 12:
    badge_html = '<span class="concession-pill tag-child">🧒 Child Concession Applied (~35% discount)</span>'
elif 18 <= age <= 24:
    badge_html = '<span class="concession-pill tag-youth">🎓 Student / Youth Saver (~10% discount)</span>'
elif age >= 60:
    badge_html = '<span class="concession-pill tag-senior">👴 Senior Citizen Special (~30% discount)</span>'
else:
    badge_html = '<span class="concession-pill tag-standard">🎟️ Standard Adult Fare</span>'

# Prepare Inference Input
input_df = pd.DataFrame([{
    'Age': age,
    'City': city,
    'Theater': theater,
    'Screen_Type': screen_type,
    'Month': month_name,
    'Day': day_name,
    'Show_Time': show_time
}])

predicted_base = float(model_pipeline.predict(input_df)[0])

# GST Computation
gst_rate = 0.12 if predicted_base <= 100 else 0.18
gst_amount = predicted_base * gst_rate
total_price = round(predicted_base + gst_amount)

# Main Screen Output
st.title("🇮🇳 India Cinema Ticket Price Estimator")
st.caption("Live demographic, geographic, and seasonal predictive engine across Indian theaters.")
st.divider()

col_main, col_breakdown = st.columns([1.2, 1])

with col_main:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color: #9ca3af; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Calculated Ticket Fare</div>
            <div class="price-display">₹{total_price:,.0f}</div>
            {badge_html}
            <div style="margin-top: 15px; color: #6b7280; font-size: 0.85rem;">Inclusive of Central & State Entertainment GST</div>
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
        f"* **City & Location:** {city}\n"
        f"* **Theater Type:** {theater}\n"
        f"* **Experience Format:** {screen_type}\n"
        f"* **Schedule:** {day_name}, {booking_date.day} {month_name} ({show_time})"
    )
