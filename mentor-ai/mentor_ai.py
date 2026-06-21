""" Mentor AI """
import streamlit as st
from course_generator import generate_course
from question_agent import generate_question, generate_feedback
from progress_tracker import track_progress
from quiz_generator import generate_quiz, evaluate_quiz

# Store state
if "module_status" not in st.session_state:
    st.session_state.module_status = [False] * 5  # Tracks which modules are complete
if "current_module" not in st.session_state:
    st.session_state.current_module = 0
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "learning_history" not in st.session_state:
    st.session_state.learning_history = []
if "completed_modules" not in st.session_state:
    st.session_state.completed_modules = set()
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None      # active quiz questions
if "quiz_module" not in st.session_state:
    st.session_state.quiz_module = None         # module the quiz belongs to
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False     # whether the quiz was evaluated

# Sidebar (Left) - Skill & Level Input

with st.sidebar:
    st.title("AI-Powered Learning Mentor")
    st.sidebar.title("Learning Input")
    skill = st.sidebar.text_input("Enter the Main Skill:")
    level = st.sidebar.selectbox(
        "Select Your Level:", ["Beginner", "Intermediate", "Advanced"]
    )
    generate_button = st.button("Generate Course")
    progress_track = st.button("Track my Progress")


# Button to generate course
tab1, tab2 = st.tabs(["Course Details", "Learning History"])

# Display Course Outline

with tab1:
    if generate_button:
        if skill and level:
            with st.spinner("Generating Course..."):
                course_outline = generate_course(skill, level)
                st.session_state.course_outline = course_outline
                st.success("Course Generated Successfully!")
        else:
            st.warning("Please enter skill and level!")
    if "course_outline" in st.session_state:
        st.markdown("### Course Modules:")
        for i, module in enumerate(st.session_state.course_outline):
            with st.expander(label=module["title"]):
                st.write(module["description"])
                if st.button(
                    f"✅ Mark Module {i+1} complete",
                    disabled=(i+1 in st.session_state.completed_modules),
                ):
                    with st.spinner("Generating quiz..."):
                        st.session_state.quiz_questions = generate_quiz(skill, module)
                        st.session_state.quiz_module = module
                        st.session_state.quiz_submitted = False
                    st.session_state.current_question = generate_question(skill, module)
                    st.session_state.current_module = i + 1
                    if len(st.session_state.completed_modules) < 5:
                        st.session_state.completed_modules.add(i+1)
                    else:
                        st.error("Already all module has been completed")
                    st.success(f"Module {i+1} marked as complete!")

        if "current_question" in st.session_state:
            st.subheader("Follow-up Question:")
            st.write(st.session_state.current_question)

            # Input for User Answer
            user_answer = st.text_area("Your Answer", key="user_answer")

            # Submit Answer Button
            if st.button("Submit Answer"):
                if user_answer.strip():
                    feedback = generate_feedback(
                        st.session_state.current_question, user_answer
                    )

                    # Store in Learning History
                    st.session_state.learning_history.append(
                        {
                            "module": st.session_state.course_outline[
                                st.session_state.current_module
                            ],
                            "question": st.session_state.current_question,
                            "answer": user_answer,
                            "feedback": feedback,
                        }
                    )

                    # Display Feedback
                    st.subheader("✅ AI Feedback:")
                    st.write(feedback)

                    # Unlock next module
                    st.session_state.current_module += 1
                else:
                    st.warning("Please enter an answer before submitting.")

        # --- Quiz Generator Agent ---
        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.divider()
            st.subheader(f"📝 Module Quiz: {st.session_state.quiz_module['title']}")
            user_answers = {}
            for idx, q in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Q{idx+1}. {q['question']}**")
                options = [f"{k}. {v}" for k, v in q["options"].items()]
                choice = st.radio(
                    label="Select your answer:",
                    options=options,
                    key=f"quiz_q_{idx}",
                    label_visibility="collapsed",
                )
                if choice:
                    user_answers[idx] = choice[0]   # store just the letter

            if st.button("Submit Quiz"):
                if len(user_answers) == len(st.session_state.quiz_questions):
                    with st.spinner("Evaluating quiz..."):
                        result = evaluate_quiz(st.session_state.quiz_questions, user_answers)
                    st.session_state.quiz_submitted = True

                    # Display results
                    st.subheader(f"Quiz Result: {result['score']}/{result['total']}")
                    for d in result["details"]:
                        icon = "✅" if d["is_correct"] else "❌"
                        st.write(
                            f"{icon} **{d['question']}** — "
                            f"Your answer: **{d['selected']}** | Correct: **{d['correct']}**"
                        )
                    st.info(f"**AI Feedback:** {result['feedback']}")

                    # Store quiz result in learning history for Progress Tracker
                    st.session_state.learning_history.append({
                        "module": st.session_state.quiz_module,
                        "question": f"[Quiz] {st.session_state.quiz_module['title']} "
                                    f"({result['score']}/{result['total']} correct)",
                        "answer": ", ".join(
                            f"Q{i+1}:{d['selected']}" for i, d in enumerate(result["details"])
                        ),
                        "feedback": result["feedback"],
                    })
                    st.session_state.quiz_questions = None
                else:
                    st.warning("Please answer all questions before submitting.")

        # Final course completion button
        if st.button("Complete Course"):
            st.session_state.learning_history.append("Course Completed 🎉")
            st.success("You have successfully completed the course!")

    else:
        st.write("🎇Generate the course by filling skill and level")
with tab2:
    st.header("Learning History")
    if st.session_state.learning_history:
        for entry in st.session_state.learning_history:
            if isinstance(entry, str):
                st.info(entry)
            else:
                with st.expander(f"**Q:** {entry['question']}"):
                    st.write(f"**Q:** {entry['question']}")
                    st.write(f"**Your A:** {entry['answer']}")
                    st.write(f"**AI Feedback:** {entry['feedback']}")
                    st.write("---")
    else:
        st.write("No progress yet.")

if progress_track:
    with st.spinner("Generating Progress..."):
        track_progress(st.session_state.learning_history, st.session_state.completed_modules)
