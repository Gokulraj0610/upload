from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import boto3
import os
import io
import logging
import json
from PIL import Image
import concurrent.futures
import threading
from queue import Queue
import numpy as np
from functools import lru_cache
import time
import math
from typing import List, Dict, Any

# Initialize Flask app
app = Flask(__name__)
CORS(app)

STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.readonly']
MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB
REKOGNITION_MAX_SIZE = 5 * 1024 * 1024  # 5MB AWS limit
BATCH_SIZE = 5  # Size of each batch
NUM_PARALLEL_BATCHES = 4  # Number of batches to process in parallel
CACHE_TTL = 3600

# Initialize cache
class ImageCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < CACHE_TTL:
                    return data
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            self.cache[key] = (value, time.time())

image_cache = ImageCache()

# Thread-local storage
thread_local = threading.local()

class OptimizedImageProcessor:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.process_queue = Queue()
        self.results = []
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.processed_hashes = set()
        
    @staticmethod
    def optimize_image(img_bytes, target_size=(800, 800), quality=85):
        try:
            img = Image.open(io.BytesIO(img_bytes))
            
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
            aspect = img.size[0] / img.size[1]
            if aspect > 1:
                new_size = (target_size[0], int(target_size[1] / aspect))
            else:
                new_size = (int(target_size[0] * aspect), target_size[1])
            
            img = img.resize(new_size, Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True, progressive=True)
            return output.getvalue()
            
        except Exception as e:
            raise Exception(f"Image optimization failed: {str(e)}")

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_image_hash(image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('L').resize((8, 8), Image.LANCZOS)
        pixels = np.array(img.getdata(), dtype=np.float64).reshape((8, 8))
        dct = np.abs(np.fft.dct(np.fft.dct(pixels, axis=0), axis=1))
        return hash(dct.tobytes())

def get_drive_service():
    if not hasattr(thread_local, "drive_service"):
        credentials = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
        thread_local.drive_service = build('drive', 'v3', credentials=credentials)
    return thread_local.drive_service

def list_drive_images():
    try:
        query = f"'{DRIVE_FOLDER_ID}' in parents and trashed = false and (mimeType contains 'image/')"
        results = get_drive_service().files().list(
            q=query,
            fields="files(id, name, mimeType, createdTime, size)",
            orderBy="createdTime desc",
            pageSize=100
        ).execute()
        return results.get('files', [])
    except Exception as e:
        logger.error(f"Error listing Drive images: {e}")
        return []

def process_single_comparison(source_bytes, drive_file, processor):
    try:
        fh = io.BytesIO()
        request = get_drive_service().files().get_media(fileId=drive_file['id'])
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        target_bytes = fh.getvalue()
        optimized_target = processor.optimize_image(target_bytes)
        
        comparison = rekognition_client.compare_faces(
            SourceImage={'Bytes': source_bytes},
            TargetImage={'Bytes': optimized_target},
            SimilarityThreshold=70
        )
        
        if comparison['FaceMatches']:
            return {
                'filename': drive_file['name'],
                'file_id': drive_file['id'],
                'similarity': comparison['FaceMatches'][0]['Similarity']
            }
            
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return None
    finally:
        if 'fh' in locals():
            fh.close()
    
    return None

def stream_results(generator):
    """Helper function to stream JSON responses"""
    for item in generator:
        yield json.dumps(item) + '\n'

def process_batch(batch, optimized_source, processor, progress_queue):
    """Process a batch of images and return results"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        future_to_file = {
            executor.submit(
                process_single_comparison,
                optimized_source,
                drive_file,
                processor
            ): drive_file for drive_file in batch
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            progress_queue.put(1)  # Increment progress counter
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error in batch comparison: {e}")
                continue
                
    return results

def process_image_stream(image_bytes):
    """Generator function to process images in parallel batches"""
    processor = OptimizedImageProcessor()
    optimized_source = processor.optimize_image(image_bytes)
    progress_queue = Queue()
    
    try:
        source_response = rekognition_client.detect_faces(
            Image={'Bytes': optimized_source},
            Attributes=['DEFAULT']
        )

        if not source_response['FaceDetails']:
            yield {'type': 'error', 'message': 'No face detected in the uploaded image'}
            return

        drive_images = list_drive_images()
        total_images = len(drive_images)
        processed = 0
        
        # Split images into batches for parallel processing
        all_batches = [
            drive_images[i:i + BATCH_SIZE]
            for i in range(0, len(drive_images), BATCH_SIZE)
        ]
        
        # Process batches in groups of NUM_PARALLEL_BATCHES
        for i in range(0, len(all_batches), NUM_PARALLEL_BATCHES):
            current_batches = all_batches[i:i + NUM_PARALLEL_BATCHES]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL_BATCHES) as executor:
                # Submit all batches for parallel processing
                future_to_batch = {
                    executor.submit(
                        process_batch,
                        batch,
                        optimized_source,
                        processor,
                        progress_queue
                    ): batch for batch in current_batches
                }
                
                # Track progress and yield results
                pending_futures = set(future_to_batch.keys())
                
                while pending_futures:
                    # Check progress queue
                    while not progress_queue.empty():
                        progress_queue.get()
                        processed += 1
                        if processed % 5 == 0:
                            yield {
                                'type': 'progress',
                                'processed': processed,
                                'total': total_images
                            }
                    
                    # Check for completed futures
                    done_futures, pending_futures = concurrent.futures.wait(
                        pending_futures,
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done_futures:
                        try:
                            batch_results = future.result()
                            for result in batch_results:
                                yield {
                                    'type': 'match',
                                    'data': result
                                }
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                            continue

    except Exception as e:
        logger.error(f"Error in face comparison: {e}")
        yield {'type': 'error', 'message': str(e)}

# Routes
@app.route('/')
def home():
    return send_file('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/app.js')
def serve_js():
    return send_from_directory('.', 'app.js')

@app.route('/styles.css')
def serve_css():
    return send_from_directory('.', 'styles.css')

@app.route('/set-folder-id', methods=['POST'])
def set_folder_id():
    global DRIVE_FOLDER_ID
    data = request.json
    DRIVE_FOLDER_ID = data.get('folderId')
    try:
        folder_metadata = get_drive_service().files().get(fileId=DRIVE_FOLDER_ID, fields="name").execute()
        return jsonify({'success': True, 'folderName': folder_metadata.get('name')})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/search-face', methods=['POST'])
def search_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        image_bytes = file.read()
        if len(image_bytes) > MAX_IMAGE_SIZE:
            return jsonify({'error': 'Image size must be less than 15MB'}), 400

        return Response(
            stream_results(process_image_stream(image_bytes)),
            mimetype='application/x-ndjson'
        )

    except Exception as e:
        logger.error(f"Error in search_face: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<file_id>')
def serve_image(file_id):
    try:
        cached_image = image_cache.get(f"img_{file_id}")
        if cached_image:
            return send_file(
                io.BytesIO(cached_image),
                mimetype='image/jpeg'
            )

        request = get_drive_service().files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        image_bytes = fh.getvalue()

        try:
            processor = OptimizedImageProcessor()
            optimized_image = processor.optimize_image(
                image_bytes,
                target_size=(1200, 1200),
                quality=90
            )
            
            image_cache.set(f"img_{file_id}", optimized_image)
            
            return send_file(
                io.BytesIO(optimized_image),
                mimetype='image/jpeg'
            )
        except Exception as e:
            logger.warning(f"Image optimization failed, serving original: {e}")
            return send_file(fh, mimetype='image/jpeg')
            
    except Exception as e:
        logger.error(f"Error serving image {file_id}: {e}")
        return jsonify({'error': str(e)}), 404
    finally:
        if 'fh' in locals():
            try:
                fh.close()
            except:
                pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize services
try:
    credentials = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    
    aws_creds = json.load(open('credentials.json'))['aws']
    rekognition_client = boto3.client('rekognition',
        aws_access_key_id=aws_creds['aws_access_key_id'],
        aws_secret_access_key=aws_creds['aws_secret_access_key'],
        region_name=aws_creds['region']
    )
    
    DRIVE_FOLDER_ID = None
    
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    raise

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False,ssl_context=('certs/cert.pem', 'certs/key.pem'))
