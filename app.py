

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# CELL CONSTELLATION
# Microscope Analysis + 50-Year Simulation
# ==========================================

st.set_page_config(
    page_title=" ASNIKA'S CELL CONSTELLATION",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬  ASNIKA'S CELL CONSTELLATION")
st.subheader("AI-Assisted Cellular Image Analysis & Future Simulation")

st.write(
    "Explore microscopic image regions, measure their "
    "properties and simulate hypothetical changes over 50 years."
)

st.warning(
    "Educational research prototype. The simulation is mathematical, "
    "not a real biological prediction or medical diagnosis."
)

# ==========================================
# SECTION 1: IMAGE UPLOAD
# ==========================================

st.header("1. MICROSCOPE IMAGE CAPTURE")

uploaded_file = st.file_uploader(
    "Upload a microscope image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff","webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.image(
        image_array,
        caption="Original microscope image",
        use_container_width=True
    )

    # ======================================
    # SECTION 2: IMAGE PROCESSING
    # ======================================

    st.header("2. CELL REGION DETECTION")

    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    threshold = st.slider(
        "Adjust image detection threshold",
        0, 255, 120
    )

    _, binary = cv2.threshold(
        blurred,
        threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result_image = image_array.copy()
    measurements = []

    region_number = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 30:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
        ) / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(contour)

        region_number += 1

        cv2.drawContours(
            result_image,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            result_image,
            str(region_number),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

        measurements.append({
            "Region": region_number,
            "Area": round(area, 2),
            "Perimeter": round(perimeter, 2),
            "Circularity": round(circularity, 3),
            "Width": w,
            "Height": h
        })

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            result_image,
            caption="Detected boundaries",
            use_container_width=True
        )

    with col2:
        st.image(
            binary,
            caption="Binary detection map",
            use_container_width=True
        )

    # ======================================
    # SECTION 3: IMAGE ANALYSIS
    # ======================================

    st.header("3. CELLULAR REGION ANALYSIS")

    if measurements:

        df = pd.DataFrame(measurements)

        average_area = df["Area"].mean()
        average_circularity = df["Circularity"].mean()

        a, b, c = st.columns(3)

        a.metric("Detected regions", len(df))
        b.metric("Average area", f"{average_area:.2f} px²")
        c.metric("Average circularity", f"{average_circularity:.3f}")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download analysis report",
            csv,
            "cell_constellation_report.csv",
            "text/csv"
        )

        st.caption(
            "Detected regions are not necessarily individual cells. "
            "Measurements are in pixels and require calibration "
            "for physical dimensions."
        )

        # ==================================
        # SECTION 4: 50-YEAR SIMULATION
        # ==================================

        st.header("4. MULTI-GENERATIONAL CELL SIMULATOR")

        st.write(
            "Explore how a hypothetical cellular health index "
            "might change over 50 simulated years."
        )

        st.info(
            "This model illustrates mathematical scenarios. "
            "It does not forecast the future of the uploaded cell."
        )

        starting_health = st.slider(
            "Starting hypothetical health index",
            1, 100, 85
        )

        scenario = st.selectbox(
            "Choose a simulation scenario",
            [
                "Stable conditions",
                "Gradual decline",
                "Increased stress",
                "Recovery scenario"
            ]
        )

        # Scenario parameters are illustrative,
        # not experimentally validated biological rates.

        if scenario == "Stable conditions":
            yearly_change = -0.1
            variability = 0.5

        elif scenario == "Gradual decline":
            yearly_change = -0.8
            variability = 0.7

        elif scenario == "Increased stress":
            yearly_change = -1.5
            variability = 1.0

        else:
            yearly_change = 0.6
            variability = 0.5

        if st.button("SIMULATE 50 YEARS"):

            np.random.seed(42)

            health = float(starting_health)
            years = []
            health_values = []

            for year in range(51):

                years.append(year)
                health_values.append(health)

                change = np.random.normal(
                    yearly_change,
                    variability
                )

                health = np.clip(
                    health + change,
                    0,
                    100
                )

            simulation_df = pd.DataFrame({
                "Year": years,
                "Hypothetical health index": health_values
            })

            st.success("50-year mathematical simulation completed!")

            # Graph

            fig, ax = plt.subplots()

            ax.plot(
                years,
                health_values,
                marker="o"
            )

            ax.set_title("50-Year Hypothetical Health Index")
            ax.set_xlabel("Simulated years")
            ax.set_ylabel("Health index (0-100)")
            ax.set_ylim(0, 100)
            ax.grid(True)

            st.pyplot(fig)

            # Milestone results

            st.subheader("Future timeline")

            for milestone in [0, 10, 20, 30, 40, 50]:

                value = health_values[milestone]

                st.write(
                    f"Year {milestone}: "
                    f"Hypothetical index = {value:.2f}/100"
                )

            st.dataframe(
                simulation_df,
                use_container_width=True
            )

            st.download_button(
                "Download 50-year simulation",
                simulation_df.to_csv(index=False).encode("utf-8"),
                "50_year_simulation.csv",
                "text/csv"
            )

    else:

        st.warning(
            "No regions detected. Try adjusting the threshold."
        )

else:

    st.info(
        "Upload a microscope image to begin your analysis."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CELL CONSTELLATION | Biology + Image Processing + "
    "Mathematical Simulation"
)