import streamlit as st
import requests


st.set_page_config(
    page_title="Vehicle Fault Diagnosis Agent",
    page_icon=None,
    layout="centered"
)


st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">Vehicle Fault Diagnosis</div>',
    unsafe_allow_html=True
)


question = st.text_area(
    "Describe the vehicle problem",
    placeholder=(
        "Example: My transmission slips when "
        "I accelerate..."
    ),
    height=130
)


if st.button(
    "Diagnose Vehicle Fault",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please describe a vehicle problem."
        )

    else:

        with st.spinner(
            "Analyzing vehicle fault..."
        ):

            try:

                response = requests.post(
                    "http://localhost:8000/diagnose",
                    json={
                        "question": question
                    },
                    timeout=60
                )

                response.raise_for_status()

                data = response.json()

                st.markdown(
                    '<div class="section-title">'
                    'Diagnosis'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    data["diagnosis"]
                )

                st.markdown(
                    '<div class="section-title">'
                    'System'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    "Pipeline: Streamlit → FastAPI → "
                    "LangGraph → Pinecone → OpenAI"
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure FastAPI is running on port 8000."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The request timed out. "
                    "Please try again."
                )

            except requests.exceptions.RequestException as e:

                st.error(
                    f"API request failed: {e}"
                )