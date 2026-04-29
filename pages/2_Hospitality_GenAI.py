from pathlib import Path
import sys
from uuid import uuid4

import streamlit as st


GENAI_ROOT = Path(__file__).resolve().parent.parent / "integrated" / "genai_module"
if str(GENAI_ROOT) not in sys.path:
    sys.path.insert(0, str(GENAI_ROOT))

from config.config import IMAGE_DIR
from database.db import (
    authenticate_user,
    create_user,
    get_relevant_history,
    get_user_generations,
    init_db,
    save_generation,
)
from services.gemini_service import generate_text
from services.image_service import generate_image
from utils.utils import validate_prompt


def save_image_file(image_bytes: bytes) -> str:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGE_DIR / f"generation_{uuid4().hex}.jpg"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def initialize_session() -> None:
    if "genai_user" not in st.session_state:
        st.session_state.genai_user = None


def render_auth_screen() -> None:
    st.title("Hospitality Concept Visualizer")
    st.write("Sign up or log in to create and save hospitality concept designs.")

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        with st.form("genai_login_form"):
            username = st.text_input("Username", key="genai_login_username")
            password = st.text_input("Password", type="password", key="genai_login_password")
            login_clicked = st.form_submit_button("Login")

        if login_clicked:
            if not username.strip() or not password.strip():
                st.error("Please enter both username and password.")
            else:
                success, user = authenticate_user(username, password)
                if success:
                    st.session_state.genai_user = user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with signup_tab:
        with st.form("genai_signup_form"):
            username = st.text_input("Choose a username", key="genai_signup_username")
            password = st.text_input("Choose a password", type="password", key="genai_signup_password")
            signup_clicked = st.form_submit_button("Create Account")

        if signup_clicked:
            if not username.strip() or not password.strip():
                st.error("Please enter both username and password.")
            else:
                success, message = create_user(username, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def render_generator(user: dict) -> None:
    st.title("Hospitality Concept Visualizer")
    st.write(f"Logged in as `{user['username']}`")

    with st.sidebar:
        st.header("Hospitality Account")
        st.write(f"User: `{user['username']}`")
        if st.button("Logout"):
            st.session_state.genai_user = None
            st.rerun()

    page = st.sidebar.radio("Navigation", ["Generate Design", "My Designs"])

    if page == "Generate Design":
        prompt = st.text_input(
            "Enter a hospitality prompt",
            placeholder="Luxury beach resort with sunset view",
        )

        if st.button("Generate"):
            is_valid, error_message = validate_prompt(prompt)
            if not is_valid:
                st.error(error_message)
            else:
                with st.spinner("Generating concept..."):
                    try:
                        personal_context = get_relevant_history(user["id"], prompt)
                        text_result = generate_text(prompt, personal_context)
                        image_result = generate_image(prompt)
                        image_path = save_image_file(image_result)
                        save_generation(user["id"], prompt, text_result, image_path)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Generated Text")
                            st.write(text_result)
                        with col2:
                            st.subheader("Generated Image")
                            st.image(image_result, caption=prompt, use_container_width=True)
                        st.success("Design saved to your history.")
                    except Exception as exc:
                        st.error(str(exc))
    else:
        st.header("My Designs")
        records = get_user_generations(user["id"])

        if not records:
            st.info("No saved designs yet. Generate a concept to see it here.")
            return

        for record in records:
            st.subheader(record["prompt"])
            st.caption(f"Created at: {record['created_at']}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(record["generated_text"])
            with col2:
                image_path = Path(record["image_path"])
                if image_path.exists():
                    image_bytes = image_path.read_bytes()
                    st.image(image_bytes, caption=record["prompt"], use_container_width=True)
                    st.download_button(
                        label="Download Image",
                        data=image_bytes,
                        file_name=f"{record['prompt'].replace(' ', '_')}.jpg",
                        mime="image/jpeg",
                        key=f"download_{record['id']}",
                    )
                else:
                    st.warning("Saved image file was not found on disk.")
            st.divider()


init_db()
initialize_session()

if st.session_state.genai_user:
    render_generator(st.session_state.genai_user)
else:
    render_auth_screen()
