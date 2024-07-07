from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, g, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from io import BytesIO
import os
import uuid
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from datetime import datetime, timedelta
import time
import fitz
import threading
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///file_db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'flask_session:'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=1)  # Set session lifetime to 15 minutes

Session(app)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Ensuring the files directory 
FILES_DIR = './files'
os.makedirs(FILES_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")
# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # This will store the hashed password
    is_admin = db.Column(db.Boolean, default=False)

# Define File model
class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('files', lazy=True))
    num_rows = db.Column(db.Integer, nullable=False)
    tokens = db.Column(db.Integer, nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    tool_used = db.Column(db.String(225), nullable=False)

# Creating admin views
class MyModelView(ModelView):
    def is_accessible(self):
        return 'username' in session and session['is_admin']

class FileModelView(MyModelView):
    column_list = ('filename', 'user_id', 'num_rows', 'tokens', 'file_size', 'processed_at')

class UserModelView(MyModelView):
    column_list = ('username', 'is_admin')
    form_columns = ('username', 'password', 'is_admin')

admin = Admin(app, name='Admin', template_mode='bootstrap3')
admin.add_view(UserModelView(User, db.session))
admin.add_view(FileModelView(File, db.session))

with app.app_context():
    db.create_all()
    
    # Check if the admin user already exists
    if not User.query.filter_by(username='admin').first():
        admin_user = User(username='admin', password=generate_password_hash('admin'), is_admin=True)
        db.session.add(admin_user)
        db.session.commit()
        print("Admin user created with default credentials.")

model = BertForSequenceClassification.from_pretrained('./saved_model')
print("Model loaded successfully.")
tokenizer = BertTokenizer.from_pretrained('./saved_tokenizer')
print("Tokenizer loaded successfully.")

def preprocess_text(text):
    # Remove URLs
    print("preprocessing started")
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove unnecessary white spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and lemmatize
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.text not in STOP_WORDS and not token.is_punct]
    print("Preprocessing done")
    return " ".join(tokens)

def extractive_summary(text, num_sentences=2):
    # Ensure sentences are properly segmented
    print("extractive summary started")
    sentences = [sent.text.strip() for sent in nlp(text).sents if len(sent.text.strip()) > 0]

    if len(sentences) <= num_sentences:
        return " ".join(sentences)  # If fewer sentences than required, return all

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf_matrix)
    sentence_scores = similarity_matrix.sum(axis=1)

    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores, axis=0)[::-1]]
    summary = " ".join(ranked_sentences[:num_sentences])
    print("Extracted summary end")
    return summary

def extract_email_data_from_pdf(pdf_file):
    # Extract text from PDF
    print("Extract email data from pdf dtarted")
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()

    # Regex for extracting data
    pattern = r'Thread ID: (.+?)\nSubject: (.+?)\nTimestamp: (.+?)\nFrom: (.+?)\nTo: (.+?)\nBody:\n(.+?)(?=\nThread ID:|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    # Create a list of dictionaries to hold email data
    email_data = []
    for match in matches:
        email_data.append({
            'thread_id': match[0].strip(),
            'subject': match[1].strip(),
            'timestamp': match[2].strip(),
            'from': match[3].strip(),
            'to': match[4].strip(),
            'body': match[5].strip()
        })
    print("Extraction end")
    return email_data

def summarize_email_chain(email_chain):
    print("Summarising email chain")
    full_body = " ".join([email['body'] for email in email_chain])
    cleaned_email = preprocess_text(full_body)
    summary = extractive_summary(cleaned_email)
    return summary

def summarize_emails_and_save_to_excel(input_pdf):
    email_data = extract_email_data_from_pdf(input_pdf)
    email_df = pd.DataFrame(email_data)
    grouped = email_df.groupby('thread_id')
    summaries = []
    for thread_id, group in grouped:
        summary = summarize_email_chain(group.to_dict('records'))
        summaries.append({
            'thread_id': thread_id,
            'summary': summary
        })
    output_excel = os.path.join(FILES_DIR, f"processed_{os.path.basename(input_pdf)}.xlsx")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_excel(output_excel, index=False)
    print(f"Summary saved as Excel file: {output_excel}")
    return output_excel

@app.route('/')
def home():
    is_admin = session.get('is_admin', False)
    return render_template('home.html', is_admin = is_admin)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash('You must be logged in to upload files.', 'danger')
        return redirect(url_for('login'))

    if 'files' not in request.files:
        return jsonify({"error": "No files part in request"}), 400

    uploaded_files = request.files.getlist('files')
    processed_files = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({"error": "Filename is undefined or empty"}), 400

        try:
            df = pd.read_excel(file)
        except Exception as e:
            return jsonify({'error': f"Error reading Excel file: {e}"}), 400

        if 'email' not in df.columns:
            return jsonify({'error': "'email' column not found in the uploaded file"}), 400

        predicted_labels = []
        confidences = []
        total_tokens = 0

        for index, row in df.iterrows():
            email_text = row['email']

            inputs = tokenizer(email_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            total_tokens += len(inputs['input_ids'][0])

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                predicted_label = torch.argmax(predictions, dim=1).item()

            label_map = {0: 'spam', 1: 'ham'}
            predicted_label_name = label_map[predicted_label]

            predicted_labels.append(predicted_label_name)
            confidences.append(predictions[0][predicted_label].item())

        df['predicted_label'] = predicted_labels
        df['confidence'] = confidences

        file_uuid = str(uuid.uuid4())
        output_filename = f"{file_uuid}_{filename}"
        output_path = os.path.join(FILES_DIR, output_filename)
        df.to_excel(output_path, index=False)

        file_size = os.path.getsize(output_path)
        num_rows = df.shape[0]
        user_id = session.get('user_id')
        if user_id:
            file_record = File(filename=output_filename, user_id=user_id, num_rows=num_rows, tokens=total_tokens, file_size=file_size, processed_at=datetime.utcnow(), tool_used="Email Classifier")
            db.session.add(file_record)
            db.session.commit()

        processed_files.append((file_uuid, output_filename))

    filenames = [f"{file_uuid}_{os.path.basename(f)}" for file_uuid, f in processed_files]
    return jsonify({'filenames': filenames}), 200

@app.route('/summarize_emails', methods=['POST'])
def summarize_emails_route():
    if 'user_id' not in session:
        flash('You must be logged in to upload files.', 'danger')
        return redirect(url_for('login'))

    if 'files' not in request.files:
        return jsonify({'error': "No files part in request"}), 400

    uploaded_files = request.files.getlist('files')
    processed_files = []

    for file in uploaded_files:
        original_filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{original_filename}"
        
        try:
            input_pdf = os.path.join(FILES_DIR, unique_filename)
            file.save(input_pdf)
            print(f"File saved: {input_pdf}")

            output_excel = summarize_emails_and_save_to_excel(input_pdf)
            os.remove(input_pdf)
            print(f"Temporary file removed: {input_pdf}")

            processed_files.append(output_excel)
        except Exception as e:
            print(f"Error processing file {original_filename}: {e}")
            return jsonify({'error': f"Error processing file: {e}"}), 400

        user_id = session.get('user_id')
        file_size = os.path.getsize(output_excel)
        if user_id:
            file_record = File(filename=unique_filename, user_id=user_id, num_rows="00", tokens="00", file_size=file_size, processed_at=datetime.utcnow(), tool_used="Email Summarizer")
            db.session.add(file_record)
            db.session.commit()

    if not processed_files:
        return jsonify({'error': "No files processed"}), 400

    filenames = [os.path.basename(f) for f in processed_files]
    print(f"Processed files: {processed_files}")
    print(f"Returning filenames: {filenames}")
    
    return jsonify({'filenames': filenames}), 200


@app.route('/download', methods=['GET'])
def download():
    filename = request.args.get('filename')
    if not filename:
        return "Filename is required", 400

    normalized_filename = os.path.normpath(filename)
    file_path = os.path.join(FILES_DIR, normalized_filename)

    if not os.path.exists(file_path):
        return "File not found", 404

    try:
        return send_from_directory(FILES_DIR, normalized_filename, as_attachment=True)
    finally:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def delete_file_after_delay(file_path, delay_seconds=10):
    time.sleep(delay_seconds)
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


@app.route('/delete', methods=['GET'])
def delete_file():
    filename = request.args.get('filename')

    if not filename:
        return "Filename is required", 400

    normalized_filename = os.path.normpath(filename)
    file_path = os.path.join(FILES_DIR, normalized_filename)

    if not os.path.exists(file_path):
        return "File not found", 404


    deletion_thread = threading.Thread(target=delete_file_after_delay, args=(file_path,))
    deletion_thread.start()

    return jsonify({'message': f'Deleting file {filename} after a short delay'}), 200

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/option1')
def option1():
    return render_template('option1.html')

@app.route('/option2')
def option2():
    return render_template('option2.html')

@app.route('/option3')
def option3():
    return render_template('option3.html')

@app.route('/know_more')
def know_more():
    return render_template('know_more.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['username'] = username
            session['is_admin'] = user.is_admin
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('is_admin', None)
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/admin/create_user', methods=['GET', 'POST'])
def create_user():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        is_admin = 'is_admin' in request.form

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            new_user = User(username=username, password=hashed_password, is_admin=is_admin)
            db.session.add(new_user)
            db.session.commit()
            flash('User created successfully!', 'success')
            return redirect(url_for('home'))
    
    return render_template('create_user.html')

@app.route('/user_panel')
def user_panel():
    if 'username' not in session:
        flash('You need to be logged in to view this page.', 'danger')
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    user_files = File.query.filter_by(user_id=user_id).all()
    return render_template('user_panel.html', files=user_files)


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
