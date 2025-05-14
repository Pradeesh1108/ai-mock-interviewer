import requests
from io import BytesIO
import wave
from dotenv import load_dotenv
import os
import utils
import google.generativeai as genai  # Replace OpenAI with Gemini
import pyaudio
import speech_recognition as sr
import time
import json

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://localhost:5002")

# Configure Gemini API
genai.configure(api_key=API_KEY)


# Unchanged functions: synthesize_text_to_audio, read_wav_parameters, play_audio, save_wav_file, record_and_transcribe
# (These are copied from your original code and remain identical)

def synthesize_text_to_audio(message, speaker_id="p230", style_wav="", language_id=""):
    """
    Sends a TTS request and returns an in-memory BytesIO stream of the WAV audio.
    """
    url = f"{TTS_SERVER_URL}/api/tts"
    data = {
        "text": message,
        "speaker_id": speaker_id,
        "style_wav": style_wav,
        "language_id": language_id,
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"TTS request failed with status {response.status_code}: {response.text}")


def read_wav_parameters(audio_stream):
    """
    Reads the WAV parameters from the in-memory audio stream.
    """
    audio_stream.seek(0)
    with wave.open(audio_stream, "rb") as wav_file:
        params = wav_file.getparams()
    return params


def play_audio(audio_stream):
    """
    Plays audio directly from an in-memory WAV audio stream.
    """
    audio_stream.seek(0)
    p = pyaudio.PyAudio()
    with wave.open(audio_stream, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=channels,
            rate=framerate,
            output=True
        )
        chunk_size = 1024
        data = wav_file.readframes(chunk_size)
        while data:
            stream.write(data)
            data = wav_file.readframes(chunk_size)
        stream.stop_stream()
        stream.close()
    p.terminate()
    print("Audio playback complete")


def save_wav_file(audio_stream, filename="output.wav"):
    """
    Saves the in-memory WAV audio stream to a file on disk.
    """
    audio_stream.seek(0)
    with open(filename, "wb") as f:
        f.write(audio_stream.read())
    print(f"Audio saved to {filename}")


def record_and_transcribe():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0
    recognizer.non_speaking_duration = 1.0
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... Speak now!")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return ""
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return ""


# Modified functions to use Gemini 2.0 Flash
def init_cv_question_stream(cv, user_intro, client=None, model=MODEL_NAME):
    user_prompt = """You are professional talent acquisition specialist conducting an interview for an AI role.
    Your task is to start a conversation with the candidate after he introduced himself.
    You have access to the candidate's CV so you can ask him about one or more of his projects/experiences.
    The question should be short and not boring.
    The text you will generate will be read by a text-to-speech engine, so you can add vocalized text if you want.
    You should not explain the beginning of the conversation or the context of the question.
    Talk directly to the candidate.
    Be kind, nice, helpful, and professional.
    You need to keep it a natural conversation.
    You need to be human-like, and to interact with the last thing that the candidate said.
    Candidate Introduction: {intro}
    CV: {cv}

    Conversation Start: """
    formatted_user_prompt = user_prompt.format(cv=cv, intro=user_intro)

    # Use Gemini API
    model = genai.GenerativeModel(model)
    response = model.generate_content(formatted_user_prompt)
    return response.text


def stream_next_cv_question(client=None, model=MODEL_NAME, cv=None, chat_history=None):
    user_prompt = """You are professional talent acquisition specialist conducting an interview for an AI role.
    Your task is to continue the conversation with the candidate after he answered the previous question.
    Continue the conversation and do not begin a new one.
    You have access to the candidate's CV so you can ask him about one or more of his projects/experiences.
    The question should be short and not boring.
    The question should not be long!
    The text you will generate will be read by a text-to-speech engine, so you can add vocalized text if you want.
    You should not explain the beginning of the conversation or the context of the question.
    Don't repeat previous questions.
    Before asking the question, give a natural transition from the previous answer.
    Don't explain anything, and don't give any notes.
    Talk directly to the candidate.
    Be kind, nice, helpful, and professional.
    You need to keep it a natural conversation.
    Chat History: {chat_history}
    CV: {cv}

    Conversation Continuity: """
    chat_history = str(chat_history)
    formatted_user_prompt = user_prompt.format(cv=cv, chat_history=chat_history)

    # Use Gemini API
    model = genai.GenerativeModel(model)
    response = model.generate_content(formatted_user_prompt)
    return response.text


def reformulate_question(client=None, model=MODEL_NAME, question_data=None):
    """
    Reformulate a technical question to make it more conversational and suitable for an interview.
    """
    prompt = f"""You are a professional technical interviewer conducting an interview for an AI role.
    Your task is to reformulate the following technical question to make it more conversational and suitable for a verbal interview.
    The reformulated question should be clear, concise, and natural sounding when read aloud by a text-to-speech system.
    Do not change the technical content or difficulty of the question.

    Original Question: {question_data['question']}

    Topic: {question_data.get('main_subject', '')}
    Difficulty: {question_data.get('difficulty', '')}

    Please provide only the reformulated question without any additional text, explanations, or context.
    """

    # Use Gemini API
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text


def generate_evaluation_report(client=None, model=MODEL_NAME, interview_data=None):
    """
    Generate an evaluation report based on the interview questions, expected answers, and candidate's responses.
    """
    prompt = """You are an expert AI interviewer tasked with evaluating a candidate's technical interview performance.
    Based on the interview questions, expected answers, and the candidate's actual responses, provide a comprehensive evaluation report.

    Your report should include:
    1. An overall assessment of the candidate's technical knowledge
    2. Specific strengths identified during the interview
    3. Areas for improvement
    4. Detailed feedback on each question, comparing the expected answer with what the candidate provided
    5. Concrete recommendations for the candidate to improve their knowledge and interview performance
    6. Be a little bit harsh and in the same time encouraging
    7. Talk directly to the candidate
    8. Use "you" and "your" to address the candidate
    9. Be professional and respectful
    10. Provide constructive and realistic feedback

    Interview Data:
    """

    for i, item in enumerate(interview_data):
        prompt += f"\n\nQuestion {i + 1}: {item['question_data']['question']}\n"
        prompt += f"Expected Answer: {item['question_data']['answer']}\n"
        prompt += f"Candidate's Response: {item['candidate_answer']}\n"
        prompt += f"Difficulty: {item['question_data'].get('difficulty', 'N/A')}\n"
        prompt += f"Topic: {item['question_data'].get('main_subject', 'N/A')}"

    # Use Gemini API
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text


def cv_interview():
    try:
        chat_history = []
        pdf_file_path = "CV/CV_ENG_9_2025_STAGE.pdf"  # Adjust to your CV path
        cv_text = utils.extract_text_from_pdf(pdf_file_path)

        intro = """Hello dear candidate, I am Josh, your virtual voice assistant for this AI role interview.
        Please introduce yourself briefly. If you stop talking for more than 5 seconds,
        I will assume you have finished your introduction."""
        model = MODEL_NAME

        # Synthesize and play the introduction
        print("Playing introduction message...")
        audio_stream = synthesize_text_to_audio(intro)
        play_audio(audio_stream)

        # Record and transcribe user's introduction
        print("Waiting for user introduction...")
        user_intro = record_and_transcribe()
        print("User introduction:", user_intro)

        # Generate the first interview question
        print("Generating first interview question...")
        beginning = init_cv_question_stream(cv_text, user_intro, model=model)
        chat_history.append({"role": "interviewer", "content": beginning})

        # Synthesize and play the first question
        print("Playing first interview question...")
        audio_stream = synthesize_text_to_audio(beginning)
        play_audio(audio_stream)
        # Make the user answer the first question
        print("Waiting for user response to the first question...")
        first_answer = record_and_transcribe()
        chat_history.append({"role": "candidate", "content": first_answer})
        # Interactive conversation loop
        num_questions = 3
        for i in range(num_questions):
            # Generate the next interview question
            print("Generating next interview question...")
            next_question = stream_next_cv_question(model=model, cv=cv_text, chat_history=chat_history)
            chat_history.append({"role": "interviewer", "content": next_question})
            # Synthesize and play the next question
            print("Playing next interview question...")
            audio_stream = synthesize_text_to_audio(next_question)
            play_audio(audio_stream)
            # Make the user answer the next question
            print("Waiting for user response to the next question...")
            next_answer = record_and_transcribe()
            chat_history.append({"role": "candidate", "content": next_answer})
            time.sleep(1)
        print("Interview complete.")
    except Exception as e:
        print("Error during interview operation:", e)


def technical_interview():
    try:
        chat_history = []
        interview_data = []

        with open("interview_questions.json") as f:
            data = json.load(f)

        intro = """Hello dear candidate, I am Josh, your virtual voice assistant for this technical interview for an AI role.
        I'll be asking you a series of technical questions to assess your knowledge and skills.
        Please answer each question as thoroughly as you can. I'll listen until you've finished speaking.
        Let's start with a brief introduction. Please tell me about your background in AI and machine learning."""
        model = MODEL_NAME

        # Synthesize and play the introduction
        print("Playing introduction message...")
        audio_stream = synthesize_text_to_audio(intro)
        play_audio(audio_stream)

        # Record and transcribe user's introduction
        print("Waiting for user introduction...")
        user_intro = record_and_transcribe()
        print("User introduction:", user_intro)
        chat_history.append({"role": "interviewer", "content": intro})
        chat_history.append({"role": "candidate", "content": user_intro})

        # Select questions of varying difficulty
        easy_questions = [q for q in data if q.get("difficulty") == "easy"]
        medium_questions = [q for q in data if q.get("difficulty") == "medium"]
        hard_questions = [q for q in data if q.get("difficulty") == "hard"]

        # Select 2 easy, 2 medium, and 1 hard question
        selected_questions = []
        if easy_questions:
            selected_questions.extend(easy_questions[:4])
        if medium_questions:
            selected_questions.extend(medium_questions[:3])
        if hard_questions:
            selected_questions.append(hard_questions[:2])

        if not selected_questions:
            selected_questions = data[:3]

        # Ask each question and record the answer
        for i, question_data in enumerate(selected_questions):
            print(f"\nProcessing question {i + 1}/{len(selected_questions)}")

            # Reformulate the question to make it more conversational
            reformulated_question = reformulate_question(model=model, question_data=question_data)
            print(f"Reformulated question: {reformulated_question}")

            # Synthesize and play the question
            print(f"Playing question {i + 1}...")
            audio_stream = synthesize_text_to_audio(reformulated_question)
            play_audio(audio_stream)

            # Record the question in chat history
            chat_history.append({"role": "interviewer", "content": reformulated_question})

            # Record and transcribe the user's answer
            print(f"Waiting for answer to question {i + 1}...")
            user_answer = record_and_transcribe()
            print(f"User answer: {user_answer}")

            # Record the answer in chat history
            chat_history.append({"role": "candidate", "content": user_answer})

            # Store question data and answer for evaluation
            interview_data.append({
                "question_data": question_data,
                "reformulated_question": reformulated_question,
                "candidate_answer": user_answer
            })

            time.sleep(1)

        # Generate the evaluation report
        print("\nGenerating evaluation report...")
        evaluation_report = generate_evaluation_report(model=model, interview_data=interview_data)

        # Tell the candidate the interview is complete
        conclusion = "Thank you for completing the technical interview. I'm now generating an evaluation report based on your responses."
        audio_stream = synthesize_text_to_audio(conclusion)
        play_audio(audio_stream)

        # Print and save the evaluation report
        print("\n=== EVALUATION REPORT ===\n")
        print(evaluation_report)

        with open("interview_evaluation.txt", "w") as f:
            f.write(evaluation_report)
        print("\nEvaluation report saved to 'interview_evaluation.txt'")

        return chat_history, evaluation_report

    except Exception as e:
        print(f"Error during technical interview: {e}")
        return None, str(e)