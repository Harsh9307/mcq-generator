import streamlit as st
from generate_mcq import generate_mcq
from fpdf import FPDF  # Import FPDF for PDF generation
from reportlab.pdfgen import canvas
# Function to set page config to prevent reload on widget interaction
def set_page_config():
    if "reload_counter" not in st.session_state:
        st.session_state.reload_counter = 0

# Function to generate PDF using FPDF
# Function to generate PDF using ReportLab
def generate_pdf(selected_questions):
    file_path = "selected_questions.pdf"
    with canvas.Canvas(file_path) as c:
        for i, question in enumerate(selected_questions, start=1):
            c.drawString(100, 800 - i * 40, f"Question {i}: {question}")
    return file_path
# Function to create or get user session state
def get_user_session():
    if 'user' not in st.session_state:
        st.session_state.user = None
    return st.session_state.user

# Function to create or get user session state
def set_user_session(user):
    st.session_state.user = user

# Function to simulate user authentication
def authenticate(username, password):
    if username == 'user' and password == 'password':
        return True
    else:
        return False
def about_section():
    st.subheader("About Our MCQ Generation Model")
    st.markdown("✨ *Welcome to the Future of Learning*: Discover our MCQ Generation Model, a groundbreaking tool designed to redefine study methods. Built on advanced technology, it offers an efficient and interactive learning experience.")
    st.markdown("👾 *Enhance Your Study Process*: Our model transforms any text into concise summaries and generates insightful multiple-choice questions (MCQs). It's not just about saving time; it's about engaging and effective learning.")
    st.markdown("🎓 *Intelligent Summarization and MCQ Creation*: At its core, our model employs a sophisticated algorithm to extract key concepts from your text, creating focused summaries. These serve as the basis for generating MCQs that test and deepen your understanding of the material.")
    st.markdown("📈 *Tailored Learning for Everyone*: Ideal for students, professionals, and the curious, our model adapts to various texts and subjects, providing a personalized learning journey for every user.")
    st.markdown("🚀 *Maximize Your Study Efficiency*: In our fast-paced world, efficient learning is key. Our model is crafted to optimize your study time, enabling quick comprehension and effective knowledge testing.")
    st.markdown("🤝 *Join the Educational Revolution*: 'PrepU' isn't just an app; it's your gateway to interactive and limitless learning. Start your journey with us and experience a smarter way to study!💫")
def main():
    st.title("Welcome to :violet[MCQ Generator] :robot_face:")
    set_page_config()

    # Sidebar with sections for "About" and "Account"
    st.sidebar.title("Navigation")
    selected_section = st.sidebar.radio("Go to", ["About", "Account"])

    # Main content area
    if selected_section == "About":
        about_section()
        
        # Add more about information as needed
    elif selected_section == "Account":
        
        if get_user_session() is None:
            st.subheader("Account")
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                if authenticate(username, password):
                    set_user_session(username)
                    st.success(f"Logged in as {username}")
                else:
                    st.error("Invalid username or password")
        else:
            # Main MCQ generator functionality
            st.subheader("MCQ Generator")

            context = st.text_area("Enter paragraph/content here...")

            num_questions = st.number_input("Number of Questions", min_value=1, max_value=15, value=5)

            # Button to trigger MCQ generation
            if st.button("Generate Questions"):
                if context:
                    # Generate MCQs based on input text and selected radiobutton
                    generated_output = generate_mcq(context, num_questions)
                    st.markdown(generated_output, unsafe_allow_html=True)

                    # Adding question selection UI with checkboxes
                    questions = generated_output.split("<br>")
                    selected_questions = {}

                    # Download PDF button
                    if st.button("Download Selected Questions as PDF"):
                        selected_questions = [questions[i] for i, selected in selected_questions.items() if selected]
                        pdf_path = generate_pdf(selected_questions)
                        st.success(f"PDF generated and downloaded: [Download PDF]({pdf_path})")

if __name__ == '__main__':
    main()
