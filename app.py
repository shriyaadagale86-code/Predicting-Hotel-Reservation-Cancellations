import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
  model_path = "best_hotel_rf_pipeline.pkl"
  if not os.path.exists(model_path):
    st.info(
        "⚙️ Model file not found. Training model automatically on startup..."
    )
    df = pd.read_csv("Hotel Reservations.csv")
    if "Booking_ID" in df.columns:
      df.drop("Booking_ID", axis=1, inplace=True)

    X = df.drop("booking_status", axis=1)
    y = df["booking_status"].apply(lambda x: 1 if x == "Canceled" else 0)

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)

  return joblib.load(model_path)


st.title("🏨 Hotel Reservation Cancellation Predictor")
st.markdown(
    "Fill in the booking parameters below to assess cancellation risk in real"
    " time."
)

pipeline = load_pipeline()

col1, col2, col3 = st.columns(3)

with col1:
  st.subheader("📌 Booking Details")
  lead_time = st.number_input("Lead Time (Days)", min_value=0, value=85)
  market_segment = st.selectbox(
      "Market Segment Type",
      ["Online", "Offline", "Corporate", "Aviation", "Complementary"],
  )
  room_type = st.selectbox(
      "Room Type Reserved",
      [
          "Room_Type 1",
          "Room_Type 2",
          "Room_Type 3",
          "Room_Type 4",
          "Room_Type 5",
          "Room_Type 6",
          "Room_Type 7",
      ],
  )
  avg_price = st.number_input(
      "Average Price per Room ($)", min_value=0.0, value=103.5
  )

with col2:
  st.subheader("👥 Guest Profile")
  no_of_adults = st.number_input("Number of Adults", min_value=0, value=2)
  no_of_children = st.number_input("Number of Children", min_value=0, value=0)
  repeated_guest = st.selectbox(
      "Is Repeated Guest?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
  )
  no_of_prev_cancellations = st.number_input(
      "Previous Cancellations", min_value=0, value=0
  )
  no_of_prev_bookings = st.number_input(
      "Previous Non-Canceled Bookings", min_value=0, value=0
  )

with col3:
  st.subheader("🗓️ Stay & Preference")
  arrival_year = st.selectbox("Arrival Year", [2017, 2018])
  arrival_month = st.slider("Arrival Month", 1, 12, 8)
  arrival_date = st.slider("Arrival Date", 1, 31, 15)
  no_of_week_nights = st.number_input(
      "Week Nights (Mon-Fri)", min_value=0, value=2
  )
  no_of_weekend_nights = st.number_input(
      "Weekend Nights (Sat-Sun)", min_value=0, value=1
  )
  meal_plan = st.selectbox(
      "Meal Plan Type",
      ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
  )
  car_parking = st.selectbox(
      "Required Car Parking Space",
      [0, 1],
      format_func=lambda x: "Yes" if x == 1 else "No",
  )
  special_requests = st.number_input(
      "Number of Special Requests", min_value=0, max_value=5, value=0
  )

st.markdown("---")

if st.button("🔮 Predict Cancellation Risk", use_container_width=True):
  input_data = pd.DataFrame([{
      "no_of_adults": no_of_adults,
      "no_of_children": no_of_children,
      "no_of_weekend_nights": no_of_weekend_nights,
      "no_of_week_nights": no_of_week_nights,
      "type_of_meal_plan": meal_plan,
      "required_car_parking_space": car_parking,
      "room_type_reserved": room_type,
      "lead_time": lead_time,
      "arrival_year": arrival_year,
      "arrival_month": arrival_month,
      "arrival_date": arrival_date,
      "market_segment_type": market_segment,
      "repeated_guest": repeated_guest,
      "no_of_previous_cancellations": no_of_prev_cancellations,
      "no_of_previous_bookings_not_canceled": no_of_prev_bookings,
      "avg_price_per_room": avg_price,
      "no_of_special_requests": special_requests,
  }])

  prediction = pipeline.predict(input_data)[0]
  probabilities = pipeline.predict_proba(input_data)[0]
  cancel_prob = probabilities[1] * 100

  if prediction == 1:
    st.error(
        f"⚠️ **High Risk of Cancellation!**\n\nEstimated Cancellation"
        f" Probability: **{cancel_prob:.1f}%**"
    )
  else:
    st.success(
        f"✅ **Low Risk of Cancellation (Booking Confirmed)**\n\nEstimated"
        f" Cancellation Probability: **{cancel_prob:.1f}%**"
    )
