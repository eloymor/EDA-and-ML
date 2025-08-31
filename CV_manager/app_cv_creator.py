import streamlit as st
from resume_creator import ResumeCreator

st.set_page_config(page_title="Resume Creator", page_icon="📝", layout="centered")

st.title("Resume Creator")
st.caption("Fill in your details and paste your resume information. The app will generate a polished resume using your local model.")

with st.form("resume_form"):
    name = st.text_input("Name", placeholder="John Doe")
    position = st.text_input("Desired Position", placeholder="Data Scientist")
    email = st.text_input("Email", placeholder="john.doe@example.com")
    phone = st.text_input("Phone Number", placeholder="+1 555 123 4567")
    address = st.text_input("Address", placeholder="123 Main St, City, Country")
    text = st.text_area("Resume Information (Paste your details)", height=250, placeholder=(
        "Summarize your experience, skills, achievements, education, projects, etc.\n\n"
        "Example:\n- 5+ years of experience in data science...\n- Proficient in Python, SQL..."
    ))

    submitted = st.form_submit_button("Create Resume", type="primary")

# Validate and run
if submitted:
    missing = []
    for label, value in {
        "Name": name,
        "Desired Position": position,
        "Email": email,
        "Phone Number": phone,
        "Address": address,
        "Resume Information": text,
    }.items():
        if not value or not value.strip():
            missing.append(label)

    if missing:
        st.warning("Please fill in the following fields: " + ", ".join(missing))
    else:
        try:
            with st.spinner("Creating your resume..."):
                creator = ResumeCreator(
                    name=name.strip(),
                    position=position.strip(),
                    email=email.strip(),
                    phone=phone.strip(),
                    address=address.strip(),
                    text=text.strip(),
                )
                result = creator.create_resume()
            st.success("Resume generated successfully!")
            st.subheader("Result")
            # The model likely returns markdown; display nicely
            st.markdown(result)
            # Provide a download option
            st.download_button(
                label="Download as Markdown",
                data=result,
                file_name=f"resume_{name.strip().replace(' ', '_')}.md",
                mime="text/markdown",
            )
        except Exception as e:
            st.error(f"Failed to generate resume: {e}")
            st.info(
                "Make sure your local model (Ollama) is running and the model specified in resume_creator.py is available."
            )
