import os
import object_detection
from flask import Flask, request, render_template, Response, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Add secret key for sessions

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])  # Fixed typo: 'method' -> 'methods'
def upload_file():
    if request.method == "POST":
        print("POST request received")  # Debug
        print("Files in request:", request.files)  # Debug
        
        if 'file' not in request.files:
            print("No file part in request")  # Debug
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        print(f"File selected: {file.filename}")  # Debug
        
        if file.filename == '':
            print("No file selected")  # Debug
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Secure the filename to prevent directory traversal attacks
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            print(f"Saving file to: {file_path}")  # Debug
            file.save(file_path)
            
            # Check if file was saved successfully
            if os.path.exists(file_path):
                print(f"File saved successfully: {file_path}")  # Debug
                return render_template('index.html', video_path=filename, success=True)
            else:
                print(f"Failed to save file: {file_path}")  # Debug
                return jsonify({'error': 'Failed to save file'}), 500
        else:
            print(f"Invalid file type: {file.filename}")  # Debug
            return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
    
    return render_template('index.html')

@app.route('/video_feed/<video_filename>')
def video_feed(video_filename):
    """Stream processed video frames"""
    print(f"Video feed requested for: {video_filename}")  # Debug
    
    # Secure the filename
    video_filename = secure_filename(video_filename)
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    print(f"Looking for video at: {video_path}")  # Debug
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")  # Debug
        return jsonify({'error': 'Video file not found'}), 404
    
    print(f"Video file found, starting stream...")  # Debug
    
    try:
        return Response(
            object_detection.generate_frames(video_path),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print(f"Error in video_feed: {e}")
        return jsonify({'error': 'Error processing video'}), 500

@app.route('/process_video/<video_filename>')
def process_video(video_filename):
    """Process and save the entire video"""
    video_filename = secure_filename(video_filename)
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    # Create output filename
    name, ext = os.path.splitext(video_filename)
    output_filename = f"{name}_processed{ext}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    try:
        # Process the video
        object_detection.process_and_save_video(video_path, output_path)
        return jsonify({
            'success': True, 
            'message': 'Video processed successfully',
            'output_file': output_filename
        })
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Serve processed video files"""
    filename = secure_filename(filename)
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Added threaded=True for better performance
