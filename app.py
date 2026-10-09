import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# --- Page Setup ---
st.set_page_config(
    page_title="ShowTime India | BookMyShow Clone",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Theme & Styling ---
st.markdown("""
<style>
    .main { background-color: #0c1017; color: #f3f4f6; }
    .bms-header {
        background: linear-gradient(90deg, #e11d48 0%, #be123c 100%);
        padding: 18px 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .screen-curve {
        height: 10px;
        width: 70%;
        margin: 25px auto 8px auto;
        border-top: 4px solid #60a5fa;
        border-radius: 50% 50% 0 0;
        box-shadow: 0 -6px 16px rgba(96, 165, 250, 0.4);
    }
    .screen-label {
        text-align: center;
        font-size: 0.75rem;
        letter-spacing: 3px;
        color: #94a3b8;
        margin-bottom: 25px;
    }
    .ticket-card {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #334155;
        color: #f8fafc;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Catalogs & Static Data ---
INDIAN_CITIES = sorted([
    "Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad",
    "Chandigarh", "Jaipur", "Lucknow", "Kochi", "Indore", "Bhopal", "Nagpur", "Surat", "Patna",
    "Visakhapatnam", "Coimbatore", "Vadodara", "Guwahati", "Dehradun", "Mysuru", "Ranchi", "Amritsar"
])

CITY_THEATERS_MAP = {
    "Pune": ["PVR INOX Phoenix Marketcity", "PVR Pavilion Mall (SB Road)", "Cinepolis Westend (Aundh)", "E-Square Shivaji Nagar"],
    "Mumbai": ["PVR INOX Maison (BKC)", "PVR Dynamix (Juhu)", "Cinepolis Nexus Seawoods", "Maratha Mandir Cinema"],
    "Delhi NCR": ["PVR Director's Cut (Vasant Kunj)", "PVR INOX Plaza (CP)", "Wave Cinemas Noida", "Delite Cinema Daryaganj"],
    "Bengaluru": ["PVR INOX Orion Mall", "PVR Forum Koramangala", "Cinepolis Nexus Shantiniketan", "Urvashi Digital 4K"],
    "Hyderabad": ["Prasads Large Screen", "PVR INOX Next Galleria", "Cinepolis Manjeera Mall", "Sudarshan 35mm"]
}
DEFAULT_THEATERS = ["PVR Multiplex", "Cinepolis", "Miraj Cinemas", "Wave Cinemas"]

MOVIES = [
    {
        "id": "m1",
        "title": "Kalki 2898 AD",
        "genre": "Sci-Fi / Action",
        "lang": "Hindi / Telugu",
        "rating": "8.4/10",
        "runtime": "3h 01m",
        "image": "https://images.unsplash.com/photo-1534447677768-be436bb09401?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": "m2",
        "title": "Dune: Part Two",
        "genre": "Sci-Fi / Adventure",
        "lang": "English / Hindi",
        "rating": "8.8/10",
        "runtime": "2h 46m",
        "image": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": "m3",
        "title": "Fighter",
        "genre": "Action / Thriller",
        "lang": "Hindi",
        "rating": "7.9/10",
        "runtime": "2h 46m",
        "image": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=60"
    },
    {
        "id": "m4",
        "title": "Stree 2",
        "genre": "Comedy / Horror",
        "lang": "Hindi",
        "rating": "8.1/10",
        "runtime": "2h 27m",
        "image": "https://images.unsplash.com/photo-1509281373149-e957c6296406?w=500&auto=format&fit=crop&q=60"
    }
]

# --- ML Pipeline ---
@st.cache_resource(show_spinner="Bootstrapping Fare Engine...")
def get_trained_pipeline():
    np.random.seed(42)
    n = 8000
    df = pd.DataFrame({
        'Age': np.random.randint(5, 75, size=n),
        'Gender': np.random.choice(["Male", "Female", "Other"], size=n),
        'City': np.random.choice(INDIAN_CITIES, size=n),
        'Screen_Type': np.random.choice(['Standard 2D', '3D', 'IMAX', '4DX', 'Gold/Recliner'], size=n),
        'Day': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=n),
        'Show_Time': np.random.choice(['Morning', 'Matinee/Afternoon', 'Prime Evening', 'Night'], size=n)
    })
    
    price = 180.0
    price += np.where(df['City'].isin(["Mumbai", "Delhi NCR", "Bengaluru"]), 80, 25)
    price += np.where(df['Screen_Type'] == 'IMAX', 220, np.where(df['Screen_Type'] == 'Gold/Recliner', 320, 40))
    price += np.where(df['Day'].isin(['Saturday', 'Sunday']), 70, 0)
    price += np.where(df['Show_Time'] == 'Prime Evening', 50, -30)
    df['Base_Fare'] = np.clip(price + np.random.normal(0, 15, size=n), 100, 950)

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Gender', 'City', 'Screen_Type', 'Day', 'Show_Time'])],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=40, random_state=42))
    ])
    pipeline.fit(df.drop('Base_Fare', axis=1), df['Base_Fare'])
    return pipeline

model_pipeline = get_trained_pipeline()

# --- Session State Management ---
if "step" not in st.session_state:
    st.session_state.step = "movies"
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = MOVIES[0]
if "selected_seats" not in st.session_state:
    st.session_state.selected_seats = []
if "booking_confirmed" not in st.session_state:
    st.session_state.booking_confirmed = False

# --- Top Navigation Bar ---
selected_city = st.sidebar.selectbox("📍 Select City", INDIAN_CITIES, index=INDIAN_CITIES.index("Pune") if "Pune" in INDIAN_CITIES else 0)
available_theaters = CITY_THEATERS_MAP.get(selected_city, DEFAULT_THEATERS)

st.markdown(f"""
<div class="bms-header">
    <div style="font-size: 1.6rem; font-weight: 800; letter-spacing: 0.5px;">🎟️ BookMyShow <span style="font-size: 0.9rem; font-weight: 400; opacity: 0.85;">| Dynamic AI Box-Office</span></div>
    <div>📍 <b>{selected_city}</b></div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# STEP 1: MOVIE SELECTION
# -------------------------------------------------------------
if st.session_state.step == "movies":
    st.subheader("Now Showing")
    movie_cols = st.columns(4)

    for idx, movie in enumerate(MOVIES):
        with movie_cols[idx]:
            st.image(movie["image"], use_container_width=True)
            st.markdown(f"**{movie['title']}**")
            st.caption(f"{movie['genre']} • {movie['runtime']}")
            st.markdown(f"<span class='badge'>⭐ {movie['rating']}</span><span class='badge'>{movie['lang']}</span>", unsafe_allow_html=True)
            if st.button("Book Tickets", key=f"btn_{movie['id']}", use_container_width=True):
                st.session_state.selected_movie = movie
                st.session_state.step = "theaters"
                st.rerun()

# -------------------------------------------------------------
# STEP 2: THEATER, DATE & SHOWTIME
# -------------------------------------------------------------
elif st.session_state.step == "theaters":
    movie = st.session_state.selected_movie
    st.button("← Back to Movies", on_click=lambda: st.session_state.update({"step": "movies"}))

    col_meta1, col_meta2 = st.columns([1, 4])
    with col_meta1:
        st.image(movie["image"], use_container_width=True)
    with col_meta2:
        st.title(movie["title"])
        st.markdown(f"**{movie['genre']}** | **{movie['runtime']}** | **{movie['lang']}**")
        st.info(f"Audience Score: {movie['rating']}")

    st.divider()
    st.subheader("Select Date & Format")
    d_col1, d_col2 = st.columns([1, 1])
    with d_col1:
        b_date = st.date_input("Show Date", datetime.date.today(), min_value=datetime.date.today())
    with d_col2:
        screen_format = st.selectbox("Experience Format", ["Standard 2D", "3D", "IMAX", "4DX", "Gold/Recliner"])

    st.subheader("Available Cinemas & Showtimes")
    for theater in available_theaters:
        with st.container():
            st.markdown(f"#### 🏢 {theater}")
            t_cols = st.columns(4)
            slots = [("10:00 AM", "Morning"), ("01:30 PM", "Matinee/Afternoon"), ("06:45 PM", "Prime Evening"), ("10:15 PM", "Night")]
            for s_idx, (s_time, s_window) in enumerate(slots):
                with t_cols[s_idx]:
                    if st.button(f"{s_time}\n({s_window})", key=f"{theater}_{s_time}"):
                        st.session_state.booking_details = {
                            "theater": theater,
                            "date": b_date,
                            "time_str": s_time,
                            "window": s_window,
                            "format": screen_format,
                            "city": selected_city
                        }
                        st.session_state.step = "seats"
                        st.rerun()
            st.write("")

# -------------------------------------------------------------
# STEP 3: SEAT MATRIX SELECTION
# -------------------------------------------------------------
elif st.session_state.step == "seats":
    details = st.session_state.booking_details
    st.button("← Back to Theaters", on_click=lambda: st.session_state.update({"step": "seats", "selected_seats": []}))
    
    st.markdown(f"### Select Seats - {details['theater']} ({details['format']})")
    st.caption(f"🗓️ {details['date'].strftime('%a, %d %b')} | ⏰ {details['time_str']}")

    # Screen graphic
    st.markdown('<div class="screen-curve"></div><div class="screen-label">ALL EYES THIS WAY (SCREEN)</div>', unsafe_allow_html=True)

    rows = ["A", "B", "C", "D", "E"]
    cols = range(1, 9)
    seat_status = {}

    st.markdown("**Executive Seats (₹ Base)**")
    for r in rows:
        c_list = st.columns(8)
        for idx, c in enumerate(cols):
            seat_code = f"{r}{c}"
            # Pre-booked simulated seat
            is_booked = (r in ["B", "D"] and c in [3, 4])
            with c_list[idx]:
                if is_booked:
                    st.button(f"✕", key=f"seat_{seat_code}", disabled=True)
                else:
                    is_selected = seat_code in st.session_state.selected_seats
                    btn_label = f"✓ {seat_code}" if is_selected else seat_code
                    if st.button(btn_label, key=f"seat_{seat_code}"):
                        if seat_code in st.session_state.selected_seats:
                            st.session_state.selected_seats.remove(seat_code)
                        else:
                            st.session_state.selected_seats.append(seat_code)
                        st.rerun()

    st.divider()
    if st.session_state.selected_seats:
        st.success(f"Selected Seats ({len(st.session_state.selected_seats)}): {', '.join(st.session_state.selected_seats)}")
        if st.button("Proceed to F&B and Checkout →", type="primary"):
            st.session_state.step = "checkout"
            st.rerun()
    else:
        st.warning("Please click at least one available seat to proceed.")

# -------------------------------------------------------------
# STEP 4: F&B ADDONS, DYNAMIC BILLING & TICKET PASS
# -------------------------------------------------------------
elif st.session_state.step == "checkout":
    details = st.session_state.booking_details
    movie = st.session_state.selected_movie
    num_seats = len(st.session_state.selected_seats)

    st.button("← Back to Seat Selection", on_click=lambda: st.session_state.update({"step": "seats"}))
    st.title("Order Summary & Checkout")

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.subheader("🍿 Grab a Bite (Optional)")
        fnb_cols = st.columns(3)
        with fnb_cols[0]:
            popcorn = st.checkbox("Tub Popcorn (+₹290)", value=False)
        with fnb_cols[1]:
            beverage = st.checkbox("Large Pepsi (+₹180)", value=False)
        with fnb_cols[2]:
            nachos = st.checkbox("Cheese Nachos (+₹240)", value=False)

        fnb_total = (290 if popcorn else 0) + (180 if beverage else 0) + (240 if nachos else 0)

        # Passenger / Demographic Inputs for ML pricing engine
        st.subheader("Viewer Details")
        viewers = []
        for i, s_code in enumerate(st.session_state.selected_seats):
            v_cols = st.columns([1, 1, 1])
            with v_cols[0]:
                st.caption(f"Seat **{s_code}**")
            with v_cols[1]:
                age = st.number_input(f"Age", min_value=5, max_value=85, value=25, key=f"v_age_{i}")
            with v_cols[2]:
                gender = st.selectbox(f"Gender", ["Male", "Female", "Other"], key=f"v_gen_{i}")
            viewers.append({"Age": age, "Gender": gender})

    with c_right:
        # Predict dynamic fare
        eval_df = pd.DataFrame([{
            'Age': v['Age'],
            'Gender': v['Gender'],
            'City': details['city'],
            'Screen_Type': details['format'],
            'Day': details['date'].strftime("%A"),
            'Show_Time': details['window']
        } for v in viewers])

        predicted_fares = model_pipeline.predict(eval_df)
        base_ticket_total = sum(predicted_fares)
        gst = base_ticket_total * 0.18
        convenience_fee = num_seats * 35.40
        grand_total = base_ticket_total + gst + convenience_fee + fnb_total

        st.markdown(f"""
        <div class="ticket-card">
            <h3>{movie['title']} ({details['format']})</h3>
            <p style="color: #94a3b8; font-size: 0.9rem;">{details['theater']} | {details['city']}</p>
            <p><b>Date:</b> {details['date'].strftime('%a, %d %b %Y')} | <b>Time:</b> {details['time_str']}</p>
            <p><b>Seats:</b> {", ".join(st.session_state.selected_seats)}</p>
            <hr style="border: 0.5px solid #334155;">
            <div style="display: flex; justify-content: space-between;"><span>Ticket Base Fare ({num_seats}x):</span> <b>₹{base_ticket_total:,.2f}</b></div>
            <div style="display: flex; justify-content: space-between;"><span>Integrated GST (18%):</span> <b>₹{gst:,.2f}</b></div>
            <div style="display: flex; justify-content: space-between;"><span>Convenience Fee:</span> <b>₹{convenience_fee:,.2f}</b></div>
            <div style="display: flex; justify-content: space-between;"><span>Food & Beverages:</span> <b>₹{fnb_total:,.2f}</b></div>
            <hr style="border: 0.5px solid #334155;">
            <div style="display: flex; justify-content: space-between; font-size: 1.3rem;"><span>Amount Payable:</span> <b style="color: #f59e0b;">₹{grand_total:,.2f}</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.button("💳 Pay & Generate Booking Pass", type="primary", use_container_width=True):
            st.session_state.booking_confirmed = True

    if st.session_state.booking_confirmed:
        st.balloons()
        st.success("🎉 Booking Confirmed! Your m-Ticket has been issued.")
        st.code(f"""
===========================================================
               BOOKMYSHOW DIGITAL PASS                     
===========================================================
 Booking ID: BMS-IN-{np.random.randint(100000, 999999)}
 Movie     : {movie['title']} ({details['format']})
 Cinema    : {details['theater']}, {details['city']}
 Screen    : Audi 03
 Show Time : {details['date'].strftime('%d %b %Y')} at {details['time_str']}
 Seats     : {", ".join(st.session_state.selected_seats)}
 Total Paid: INR {grand_total:,.2f} (Includes GST & Fees)
===========================================================
        Please present this SMS / QR at cinema entrance.
        """, language="text")
        
        if st.button("Book Another Show", type="secondary"):
            st.session_state.step = "movies"
            st.session_state.selected_seats = []
            st.session_state.booking_confirmed = False
            st.rerun()
