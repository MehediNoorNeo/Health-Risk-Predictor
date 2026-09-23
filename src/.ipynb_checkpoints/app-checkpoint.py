from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
STYLE_PATH = Path(__file__).resolve().parent / "styles" / "style.css"

DATA_PATH = PROJECT_DIR / "data" / "novagen_dataset.csv"
MODEL_PATH = PROJECT_DIR / "models" / "stacking_pipeline.pkl"
# Health icon
ICON_PATH = BASE_DIR / "figures" / "public_health_icon.png"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HealthRisk Predictor",
    page_icon=str(ICON_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

# ============================================================
# CUSTOM CSS
# ============================================================

def load_css():
    with open(STYLE_PATH, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )


load_css()




# ============================================================
# HTML TEMPLATE HELPERS
# ============================================================

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def load_html_template(filename, replacements=None):
    template_path = TEMPLATE_DIR / filename

    if not template_path.exists():
        raise FileNotFoundError(
            f"HTML template was not found:\n{template_path}"
        )

    html = template_path.read_text(encoding="utf-8")

    if replacements:
        for key, value in replacements.items():
            html = html.replace("{{" + key + "}}", str(value))

    return html


# ============================================================
# LOAD FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset file was not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# INPUT HELPERS
# ============================================================

def category_input(
    label,
    key,
    help_text,
    default=1
):

    return st.selectbox(
        label,
        options=[0, 1],
        index=default,
        key=key,
        help=help_text,
    )


def patient_frame(values):

    return pd.DataFrame([values])


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(
    values,
    model,
    data
):
    """
    Create the input DataFrame and align its columns with the
    feature names used by the trained model whenever available.
    """

    input_df = patient_frame(values)

    expected_features = getattr(
        model,
        "feature_names_in_",
        None
    )

    if expected_features is not None:

        expected_features = list(
            expected_features
        )

        missing_features = [
            col
            for col in expected_features
            if col not in input_df.columns
        ]

        if missing_features:

            raise ValueError(
                "The application is missing model features:\n"
                + ", ".join(missing_features)
            )

        # Keep only features expected by the model
        # and preserve exact training order.

        input_df = input_df.reindex(
            columns=expected_features
        )

    else:

        # Fallback:
        # Use training dataset feature order.

        training_features = [
            col
            for col in data.columns
            if col != "Target"
        ]

        if set(input_df.columns) == set(
            training_features
        ):

            input_df = input_df[
                training_features
            ]

        else:

            raise ValueError(
                "Could not determine the feature structure "
                "expected by the trained model. Check the "
                "training pipeline and application input columns."
            )

    return input_df


# ============================================================
# POSITIVE CLASS PROBABILITY
# ============================================================

def get_positive_class_probability(
    model,
    input_df
):
    """
    Return probability of class 1.

    Class 1 represents the risk / unhealthy class.

    This avoids assuming that probability column 1 always
    corresponds to class 1.
    """

    if not hasattr(
        model,
        "predict_proba"
    ):

        raise ValueError(
            "The loaded model does not support predict_proba()."
        )

    if not hasattr(
        model,
        "classes_"
    ):

        raise ValueError(
            "The loaded model does not expose classes_."
        )

    classes = list(
        model.classes_
    )

    if 1 not in classes:

        raise ValueError(
            f"Expected binary target class 1, "
            f"but model classes are: {classes}"
        )

    positive_index = classes.index(1)

    probabilities = model.predict_proba(
        input_df
    )

    return float(
        probabilities[
            0,
            positive_index
        ]
    )


# ============================================================
# LOAD DATA + MODEL
# ============================================================

try:

    data = load_data()
    model = load_model()

except Exception as exc:

    st.error(
        "The application could not load the dataset/model."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = {

    "Target",

    # Numeric features
    "Age",
    "BMI",
    "Blood_Pressure",
    "Cholesterol",
    "Glucose_Level",
    "Heart_Rate",
    "Sleep_Hours",
    "Exercise_Hours",
    "Water_Intake",
    "Stress_Level",

    # Categorical features
    "Smoking",
    "Alcohol",
    "Diet",
    "MentalHealth",
    "PhysicalActivity",
    "MedicalHistory",
    "Allergies",

    # Diet type one-hot features
    "Diet_Type__Vegan",
    "Diet_Type__Vegetarian",

    # Blood group one-hot features
    "Blood_Group_AB",
    "Blood_Group_B",
    "Blood_Group_O",
}


missing_dataset_columns = (
    required_columns
    - set(data.columns)
)


if missing_dataset_columns:

    st.error(
        "The dataset is missing required columns: "
        + ", ".join(
            sorted(missing_dataset_columns)
        )
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.html(load_html_template("header.html"))


# ============================================================
# TABS
# ============================================================

predict_tab, insights_tab, about_tab = st.tabs(
    [
        "Risk assessment",
        "Dataset insights",
        "About",
    ]
)


# ============================================================
# RISK ASSESSMENT
# ============================================================

with predict_tab:

    st.markdown(
        '<p class="section-label">Patient profile</p>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Enter the available health indicators, then generate "
        "a prediction from the trained stacking model."
    )


    # ========================================================
    # FORM
    # ========================================================

    with st.form("risk_form"):

        clinical, lifestyle, profile = st.columns(3)


        # ====================================================
        # CLINICAL
        # ====================================================

        with clinical:

            st.markdown(
                "#### Clinical indicators"
            )

            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=120.0,
                value=45.0,
                step=1.0,
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=25.0,
                step=0.1,
            )

            blood_pressure = st.number_input(
                "Blood pressure",
                min_value=50.0,
                max_value=200.0,
                value=120.0,
                step=1.0,
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=50.0,
                max_value=500.0,
                value=200.0,
                step=1.0,
            )

            glucose = st.number_input(
                "Glucose level",
                min_value=40.0,
                max_value=500.0,
                value=100.0,
                step=1.0,
            )

            heart_rate = st.number_input(
                "Heart rate",
                min_value=30.0,
                max_value=220.0,
                value=72.0,
                step=1.0,
            )


        # ====================================================
        # LIFESTYLE
        # ====================================================

        with lifestyle:

            st.markdown(
                "#### Lifestyle indicators"
            )

            sleep = st.number_input(
                "Sleep hours",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5,
            )

            exercise = st.number_input(
                "Exercise hours",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.5,
            )

            water = st.number_input(
                "Water intake",
                min_value=0.0,
                max_value=15.0,
                value=3.0,
                step=0.5,
            )

            stress = st.slider(
                "Stress level",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=1.0,
            )

            # ------------------------------------------------
            # CATEGORICAL VARIABLES
            # Dataset uses 0, 1
            # ------------------------------------------------

            smoking = category_input(
                "Smoking",
                "smoking",
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )

            alcohol = category_input(
                "Alcohol",
                "alcohol",
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )


        # ====================================================
        # PROFILE
        # ====================================================

        with profile:

            st.markdown(
                "#### Health profile"
            )

            # ------------------------------------------------
            # CATEGORICAL VARIABLES
            # ------------------------------------------------

            diet = category_input(
                "Diet",
                "diet", 
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )

            mental_health = category_input(
                "Mental health",
                "mental_health",
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )

            activity = category_input(
                "Physical activity",
                "physical_activity",
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )

            medical_history = category_input(
                "Medical history",
                "medical_history",
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )

            allergies = category_input(
                "Allergies",
                "allergies",
                "Select the category code used in the training data: 0 or 1.",
                default=1,
            )


            # ------------------------------------------------
            # DIET TYPE
            # ------------------------------------------------

            diet_type = st.selectbox(
                "Diet type",
                options=[
                    "Vegetarian",
                    "Vegan",
                    "Other",
                ],
                index=0,
                key="diet_type",
                help=(
                    "Converted to the one-hot columns expected "
                    "by the trained model."
                ),
            )


            # ------------------------------------------------
            # BLOOD GROUP
            # ------------------------------------------------

            blood_group = st.selectbox(
                "Blood group",
                options=[
                    "A",
                    "AB",
                    "B",
                    "O",
                ],
                index=0,
                key="blood_group",
                help=(
                    "Converted to the one-hot columns expected "
                    "by the trained model."
                ),
            )


        # ====================================================
        # SUBMIT BUTTON
        # ====================================================

        submitted = st.form_submit_button(
            "Generate risk assessment"
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        # ----------------------------------------------------
        # VALIDATE CATEGORICAL SELECTIONS
        # ----------------------------------------------------

        categorical_values = {

            "Smoking": smoking,
            "Alcohol": alcohol,
            "Diet": diet,
            "MentalHealth": mental_health,
            "PhysicalActivity": activity,
            "MedicalHistory": medical_history,
            "Allergies": allergies,
            "Diet Type": diet_type,
            "Blood Group": blood_group,
        }


        missing_inputs = [
            name
            for name, value
            in categorical_values.items()
            if value is None
        ]


        if missing_inputs:

            st.error(
                "Please select all required categorical inputs: "
                + ", ".join(missing_inputs)
            )

            st.stop()


        # ----------------------------------------------------
        # DIET TYPE → ONE-HOT
        # ----------------------------------------------------

        diet_vegan = int(
            diet_type == "Vegan"
        )

        diet_vegetarian = int(
            diet_type == "Vegetarian"
        )


        # ----------------------------------------------------
        # BLOOD GROUP → ONE-HOT
        #
        # A  = 0,0,0
        # AB = 1,0,0
        # B  = 0,1,0
        # O  = 0,0,1
        # ----------------------------------------------------

        blood_ab = int(
            blood_group == "AB"
        )

        blood_b = int(
            blood_group == "B"
        )

        blood_o = int(
            blood_group == "O"
        )


        # ----------------------------------------------------
        # BUILD INPUT
        # ----------------------------------------------------

        inputs = {

            # Numeric
            "Age": age,
            "BMI": bmi,
            "Blood_Pressure": blood_pressure,
            "Cholesterol": cholesterol,
            "Glucose_Level": glucose,
            "Heart_Rate": heart_rate,
            "Sleep_Hours": sleep,
            "Exercise_Hours": exercise,
            "Water_Intake": water,
            "Stress_Level": stress,

            # Categorical
            "Smoking": smoking,
            "Alcohol": alcohol,
            "Diet": diet,
            "MentalHealth": mental_health,
            "PhysicalActivity": activity,
            "MedicalHistory": medical_history,
            "Allergies": allergies,

            # Diet type one-hot
            "Diet_Type__Vegan": diet_vegan,
            "Diet_Type__Vegetarian": diet_vegetarian,

            # Blood group one-hot
            "Blood_Group_AB": blood_ab,
            "Blood_Group_B": blood_b,
            "Blood_Group_O": blood_o,
        }


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            # ------------------------------------------------
            # Prepare input
            # ------------------------------------------------

            input_df = prepare_model_input(
                inputs,
                model,
                data,
            )


            # ------------------------------------------------
            # Predict class
            # ------------------------------------------------

            predicted_class = model.predict(
                input_df
            )[0]


            # ------------------------------------------------
            # Class 1 probability
            # ------------------------------------------------

            probability = get_positive_class_probability(
                model,
                input_df,
            )


        except Exception as exc:

            st.error(
                "Prediction failed. This usually means that "
                "the application input does not match the "
                "trained model."
            )

            st.exception(exc)

            st.stop()


        # ====================================================
        # RESULT
        #
        # Class 0 = No Risk / Healthy
        # Class 1 = Risk / Unhealthy
        # ====================================================
        
        is_positive = (
            predicted_class == 1
        )
        
        
        # ----------------------------------------------------
        # RESULT CONTENT
        # ----------------------------------------------------
        
        if is_positive:
        
            result_color = "#C73E4D"
            result_bg = "#FCECEE"
            result_border = "#F0B7BF"
            result_icon_bg = "#F8DDE1"
        
            result_label = "Health Risk Detected"
            result_short_label = "Risk / Unhealthy"
        
            status_message = (
                "The model identified a potential health-risk "
                "pattern in the provided health indicators."
            )
        
            result_icon = "!"
        
        else:
        
            result_color = "#16805C"
            result_bg = "#E8F6F0"
            result_border = "#A9DCC8"
            result_icon_bg = "#D5F0E3"
        
            result_label = "No Health Risk Detected"
            result_short_label = "Healthy / No Risk"
        
            status_message = (
                "The model did not identify a significant "
                "health-risk pattern in the provided "
                "health indicators."
            )
        
            result_icon = "✓"
        
        
        # ====================================================
        # RESULT SECTION
        # ====================================================
        
        st.divider()
        
        
        st.html(
            """
            <div style="
                margin: 0.5rem 0 1.25rem 0;
            ">
        
                <div style="
                    color: #0f766e;
                    font-size: 0.75rem;
                    font-weight: 800;
                    letter-spacing: 0.14em;
                    text-transform: uppercase;
                    margin-bottom: 0.35rem;
                ">
                    Assessment Result
                </div>
        
                <div style="
                    color: #0f172a;
                    font-size: 1.75rem;
                    font-weight: 750;
                    margin: 0;
                ">
                    Model-based health risk assessment
                </div>
        
            </div>
            """
        )
        
        
        # ====================================================
        # MAIN RESULT CARDS
        # ====================================================
        
        left, right = st.columns(
            [1.35, 1],
            gap="large"
        )
        
        
        # ====================================================
        # LEFT — STATUS CARD
        # ====================================================
        
        with left:
        
            result_html = load_html_template(
                "result_card.html",
                {
                    "result_bg": result_bg,
                    "result_border": result_border,
                    "result_icon_bg": result_icon_bg,
                    "result_color": result_color,
                    "result_icon": result_icon,
                    "result_label": result_label,
                    "result_short_label": result_short_label,
                    "status_message": status_message,
                },
            )

            st.html(result_html)
        
        
        # ====================================================
        # RIGHT — RISK PROBABILITY CARD
        # ====================================================
        
        with right:
        
            probability_percent = (
                probability * 100
            )
        
        
            # ------------------------------------------------
            # Risk level
            # ------------------------------------------------
        
            if probability < 0.30:
        
                probability_label = (
                    "Low estimated risk"
                )
        
                probability_color = (
                    "#047857"
                )
        
                probability_bg = (
                    "#ecfdf5"
                )
        
                probability_border = (
                    "#a7f3d0"
                )
        
            elif probability < 0.60:
        
                probability_label = (
                    "Moderate estimated risk"
                )
        
                probability_color = (
                    "#b45309"
                )
        
                probability_bg = (
                    "#fffbeb"
                )
        
                probability_border = (
                    "#fde68a"
                )
        
            else:
        
                probability_label = (
                    "Higher estimated risk"
                )
        
                probability_color = (
                    "#dc2626"
                )
        
                probability_bg = (
                    "#fef2f2"
                )
        
                probability_border = (
                    "#fecaca"
                )
        
        
            probability_html = load_html_template(
                "probability_card.html",
                {
                    "probability_color": probability_color,
                    "probability_bg": probability_bg,
                    "probability_border": probability_border,
                    "probability_percent": f"{probability_percent:.1f}",
                    "probability_label": probability_label,
                },
            )

            st.html(probability_html)
        
        
        # ====================================================
        # RESPONSIBLE USE
        # ====================================================
        
        st.info(
            "This is a machine-learning screening tool, not a "
            "medical diagnosis. The prediction and probability "
            "are model outputs based on the information provided "
            "and should not be used alone for medical or emergency "
            "decisions."
        )


# ============================================================
# DATASET INSIGHTS
# ============================================================

with insights_tab:

    target = pd.to_numeric(
        data["Target"],
        errors="coerce",
    )


    valid_target = target.dropna()


    # --------------------------------------------------------
    # VALIDATE TARGET
    # --------------------------------------------------------

    unique_target_values = set(
        valid_target.unique()
    )


    if not unique_target_values.issubset(
        {0, 1}
    ):

        st.error(
            "Target must contain binary values 0 and 1 "
            "for these dashboard calculations."
        )

    else:

        # Class 1 = risk
        risk_rate = target.mean()

        # Class 0 = no risk
        no_risk_rate = 1 - risk_rate

        age_avg = data["Age"].mean()

        bmi_avg = data["BMI"].mean()

        total = len(data)


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Training records",
            f"{total:,}",
        )


        c2.metric(
            "Risk-class share",
            f"{risk_rate:.1%}",
        )


        c3.metric(
            "Average age",
            f"{age_avg:.1f}",
        )


        c4.metric(
            "Average BMI",
            f"{bmi_avg:.1f}",
        )


        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        left, right = st.columns(2)


        # ====================================================
        # CLASS BALANCE
        # ====================================================

        with left:

            st.markdown(
                "#### Class balance"
            )


            class_counts = (
                target
                .dropna()
                .value_counts()
                .sort_index()
            )


            class_counts.index = [
                (
                    "Class 0 — No Risk"
                    if int(x) == 0
                    else
                    "Class 1 — Risk"
                )
                for x in class_counts.index
            ]


            st.bar_chart(
                class_counts
            )


        # ====================================================
        # CORRELATION
        # ====================================================

        with right:

            st.markdown(
                "#### Numeric relationships with Target"
            )


            correlation_columns = [

                "BMI",
                "Blood_Pressure",
                "Cholesterol",
                "Glucose_Level",
                "Stress_Level",

            ]


            correlation_df = data[
                correlation_columns
                + ["Target"]
            ].copy()


            correlation_df[
                "Target"
            ] = pd.to_numeric(
                correlation_df["Target"],
                errors="coerce",
            )


            correlations = (
                correlation_df
                .corr(
                    numeric_only=True
                )["Target"]
                .drop("Target")
                .dropna()
                .sort_values()
            )


            st.bar_chart(
                correlations
            )


# ============================================================
# ABOUT
# ============================================================

with about_tab:

    # --------------------------------------------------------
    # MODEL OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        "### Model overview"
    )


    st.write(
        "Predictions use the project's trained stacking "
        "pipeline. The pipeline combines logistic regression, "
        "random forest, and XGBoost estimators."
    )


    # --------------------------------------------------------
    # INPUT CODING
    # --------------------------------------------------------

    st.markdown(
        "### Input coding"
    )


    st.write(
        "Smoking, Alcohol, Diet, MentalHealth, "
        "PhysicalActivity, MedicalHistory, and Allergies "
        "use category codes 0 and 1 in the application "
        "to match the training data."
    )


    # --------------------------------------------------------
    # ONE-HOT FEATURES
    # --------------------------------------------------------

    st.markdown(
        "### One-hot features"
    )


    st.write(
        "Diet type and blood group are entered as categories "
        "and converted into the same one-hot columns expected "
        "by the trained model."
    )


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    st.markdown(
        "### Target"
    )


    st.html(load_html_template("target.html"))


    st.write(
        "The model predicts a binary target. Class 0 represents "
        "the no-risk / healthy class, while Class 1 represents "
        "the risk / unhealthy class. The estimated risk probability "
        "shown in the assessment represents the model's probability "
        "for Class 1."
    )


    # --------------------------------------------------------
    # RESPONSIBLE USE
    # --------------------------------------------------------

    st.markdown(
        "### Responsible use"
    )


    st.write(
        "Use this application as a data-science demonstration "
        "or supporting model signal. Do not use it for diagnosis, "
        "treatment decisions, or emergency guidance."
    )