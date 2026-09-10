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

# Custom Styling
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
        font-size: 2.8rem;
        font-weight: 800;
        color: #f59e0b;
        margin: 10px 0;
    }
    .concession-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .tag-child { background-color: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
    .tag-youth { background-color: #3b2a59; color: #c084fc; border: 1px solid #9333ea; }
    .tag-senior { background-color: #3f2e22; color: #f97316; border: 1px solid #ea580c; }
    .tag-standard { background-color: #1f2937; color: #9ca3af; border: 1px solid #4b5563; }
    </style>
""", unsafe_allow_html=True)

# Comprehensive All-India Cities List
INDIAN_CITIES = sorted([
    "Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad",
    "Chandigarh", "Jaipur", "Lucknow", "Kochi", "Indore", "Bhopal", "Nagpur", "Surat", "Patna",
    "Bhubaneswar", "Visakhapatnam", "Coimbatore", "Vadodara", "Guwahati", "Varanasi", "Dehradun",
    "Mysuru", "Agra", "Ranchi", "Amritsar", "Madurai", "Thiruvananthapuram",
    "Vijayawada", "Guntur", "Tirupati", "Warangal", "Mangaluru", "Hubballi-Dharwad",
    "Kozhikode", "Thrissur", "Kollam", "Kannur", "Tiruchirappalli", "Salem", "Tirunelveli", "Vellore",
    "Kanpur", "Prayagraj", "Meerut", "Bareilly", "Gorakhpur", "Ludhiana", "Jalandhar",
    "Jodhpur", "Udaipur", "Kota", "Gwalior", "Jabalpur", "Raipur", "Jammu", "Srinagar",
    "Nashik", "Chhatrapati Sambhaji Nagar", "Kolhapur", "Solapur", "Rajkot", "Bhavnagar", "Panaji", "Dharashiv", "Latur", "Beed",
    "Siliguri", "Asansol", "Durgapur", "Jamshedpur", "Dhanbad", "Cuttack", "Gaya", "Muzaffarpur", "Shillong"
])

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

GENDERS = ["Male", "Female", "Other"]

# Cache the trained pipeline directly inside the runtime environment
@st.cache_resource(show_spinner="Training All-India Pricing Engine...")
def get_trained_pipeline():
    np.random.seed(42)
    n_samples = 12000

    ages = np.random.randint(3, 80, size=n_samples)
    genders = np.random.choice(GENDERS, size=n_samples, p=[0.49, 0.49, 0.02])
    cities = np.random.choice(INDIAN_CITIES, size=n_samples)
    theaters = np.random.choice(THEATER_CHAINS, size=n_samples)
    screen_types = np.random.choice(
        ['Standard 2D', '3D', 'IMAX', '4DX', 'Gold/Recliner'], 
        size=n_samples, 
        p=[0.40, 0.25, 0.15, 0.08, 0.12]
    )
    months = np.random.choice(ALL_MONTHS, size=n_samples)
    days = np.random.choice(DAYS_OF_WEEK, size=n_samples)
    show_times = np.random.choice(
        ['Morning', 'Matinee/Afternoon', 'Prime Evening', 'Night'], 
        size=n_samples, 
        p=[0.2, 0.25, 0.4, 0.15]
    )

    tier_1_cities = {
        "Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", 
        "Chennai", "Kolkata", "Pune", "Ahmedabad"
    }
    tier_2_cities = {
        "Chandigarh", "Jaipur", "Lucknow", "Kochi", "Indore", "Bhopal", 
        "Nagpur", "Surat", "Visakhapatnam", "Coimbatore", "Vadodara", 
        "Kanpur", "Prayagraj", "Ludhiana", "Vijayawada", "Kozhikode", "Nashik", "Rajkot"
    }
    peak_months = {"May", "June", "October", "November", "December"}

    base_prices = []
    for age_val, gender, city, theater, screen, month, day, show in zip(ages, genders, cities, theaters, screen_types, months, days, show_times):
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
            price -= 20
            
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
            
        # Optional Demographic adjustment (e.g. promotional women-day/ladies special rebate)
        if gender == "Female" and day in ["Tuesday", "Wednesday"]:
            price *= 0.95
            
        price += np.random.normal(0, 15)
        base_prices.append(max(80.0, round(price, 2)))

    df = pd.DataFrame({
        'Age': ages,
        'Gender': genders,
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

    categorical_cols = ['Gender', 'City', 'Theater', 'Screen_Type', 'Month', 'Day', 'Show_Time']
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

# Sidebar - Show & Booking Controls
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🎬 Cinema Show Details")
    
    city = st.selectbox("Select City", INDIAN_CITIES, index=0)
    theater = st.selectbox("Theater / Chain", THEATER_CHAINS, index=0)
    
    booking_date = st.date_input(
        "Booking Date",
        value=datetime.date.today(),
        min_value=datetime.date.today(),
        max_value=datetime.date.today() + datetime.timedelta(days=365)
    )
    
    day_name = booking_date.strftime("%A")
    month_name = booking_date.strftime("%B")
    
    st.caption(f"📅 **Day:** {day_name} | **Month:** {month_name}")

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

# Main Screen Interface
st.title("🇮🇳 Multi-Ticket Box-Office Pricing Engine")
st.caption("Live dynamic fare estimator accounting for viewer demographics, formats, and peak demand.")
st.divider()

# Viewer / Ticket Setup Section
st.subheader("👥 Viewer Demographics & Quantity")
num_tickets = st.number_input("Number of Tickets", min_value=1, max_value=10, value=2, step=1)

viewer_data = []
cols = st.columns(min(int(num_tickets), 4))

for i in range(int(num_tickets)):
    col_idx = i % min(int(num_tickets), 4)
    with cols[col_idx]:
        st.markdown(f"**Ticket #{i + 1}**")
        v_age = st.number_input(f"Age", min_value=3, max_value=90, value=25, step=1, key=f"age_{i}")
        v_gender = st.selectbox(f"Gender", GENDERS, key=f"gender_{i}")
        viewer_data.append({"Age": v_age, "Gender": v_gender})

# Build inference batch dataframe
batch_records = []
for viewer in viewer_data:
    batch_records.append({
        'Age': viewer['Age'],
        'Gender': viewer['Gender'],
        'City': city,
        'Theater': theater,
        'Screen_Type': screen_type,
        'Month': month_name,
        'Day': day_name,
        'Show_Time': show_time
    })

input_df = pd.DataFrame(batch_records)

# Predictions & Tax Calculations
predicted_bases = model_pipeline.predict(input_df)

itemized_results = []
for idx, (base, viewer) in enumerate(zip(predicted_bases, viewer_data)):
    rate = 0.12 if base <= 100 else 0.18
    gst = base * rate
    total = round(base + gst)
    
    # Category tag
    age_val = viewer['Age']
    if age_val < 12:
        tag = '<span class="concession-pill tag-child">Child</span>'
    elif 18 <= age_val <= 24:
        tag = '<span class="concession-pill tag-youth">Student/Youth</span>'
    elif age_val >= 60:
        tag = '<span class="concession-pill tag-senior">Senior</span>'
    else:
        tag = '<span class="concession-pill tag-standard">Standard</span>'
        
    itemized_results.append({
        "Ticket": f"#{idx + 1} ({viewer['Gender']}, {viewer['Age']}y)",
        "Category": tag,
        "Base Fare": round(base, 2),
        "GST Rate": f"{int(rate * 100)}%",
        "GST Amount": round(gst, 2),
        "Total (INR)": total
    })

results_df = pd.DataFrame(itemized_results)
total_order_base = sum(results_df["Base Fare"])
total_order_gst = sum(results_df["GST Amount"])
grand_total = sum(results_df["Total (INR)"])

# Display Section
st.write("")
col_main, col_breakdown = st.columns([1.1, 1.2])

with col_main:
    badges_markup = "".join(results_df["Category"].tolist())
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="color: #9ca3af; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px;">Total Payable ({num_tickets} Tickets)</div>
            <div class="price-display">₹{grand_total:,.0f}</div>
            <div>{badges_markup}</div>
            <div style="margin-top: 15px; color: #6b7280; font-size: 0.85rem;">Inclusive of Central & State GST</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    with st.expander("🔍 View Batch Model Inputs"):
        st.dataframe(input_df, hide_index=True, use_container_width=True)

with col_breakdown:
    st.subheader("Invoice & Ticket Summary")
    
    display_table = results_df[["Ticket", "Base Fare", "GST Rate", "GST Amount", "Total (INR)"]].copy()
    display_table["Base Fare"] = display_table["Base Fare"].apply(lambda x: f"₹{x:.2f}")
    display_table["GST Amount"] = display_table["GST Amount"].apply(lambda x: f"₹{x:.2f}")
    display_table["Total (INR)"] = display_table["Total (INR)"].apply(lambda x: f"₹{x:.2f}")
    
    st.dataframe(display_table, hide_index=True, use_container_width=True)
    
    st.info(
        f"**Booking Overview:**\n"
        f"* **Total Base Fare:** ₹{total_order_base:.2f}\n"
        f"* **Total GST:** ₹{total_order_gst:.2f}\n"
        f"* **Format:** {screen_type} at {theater} ({city})\n"
        f"* **Slot:** {day_name}, {booking_date.day} {month_name} - {show_time}"
    )
