from flask import Flask, render_template, request, flash, redirect, url_for, session, jsonify
from flask_mail import Mail, Message
import os
import random
import string
import numpy as np
import cv2
from skimage import morphology, exposure, filters
from scipy.ndimage import binary_fill_holes
from tensorflow.keras.models import load_model
import h5py
import tempfile
import shutil
import base64
import tensorflow as tf 
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt


app = Flask(__name__)
app.secret_key = '###' # Removed for security 
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '####' # Removed for security
app.config['MAIL_PASSWORD'] = '####' # Removed for security
app.config['MAIL_USE_TLS'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True

mail = Mail(app)
model = load_model('BrainTumorClassfier2.h5')
otps = {}
# Define the dice coefficient for model evaluation
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

# Load the model once and reuse it
model_path='non_fold_unet.h5'
segmentation_model = load_model(model_path, custom_objects={'dice_coefficient': dice_coefficient})

def find_image_in_hdf5(group):
    """
    Recursively find the first suitable image in an HDF5 group.
    """
    best_candidate = None
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Dataset):
            data = np.array(item)
            if data.ndim >= 2 and data.dtype.kind in {'u', 'i', 'f'}:
                if best_candidate is None or (data.size > best_candidate.size and not np.isclose(data.max(), data.min())):
                    best_candidate = data
        elif isinstance(item, h5py.Group):
            result = find_image_in_hdf5(item)
            if result is not None:
                best_candidate = result if best_candidate is None or result.size > best_candidate.size else best_candidate
    return best_candidate

def load_mat(file_path):
    """
    Load a .mat file, extract the image for prediction, and return both the image data
    and its Base64 encoded string for display.
    """
    try:
        with h5py.File(file_path, 'r') as file:
            image_seg = find_image_in_hdf5(file)  # Extract the image data
            if image_seg is None:
                print("No suitable image dataset found in the .mat file.")
                raise ValueError("No suitable image dataset found in the .mat file.")
            
            # Ensure image_data is a numpy array and in the correct format
            if not isinstance(image_seg, np.ndarray):
                image_seg = np.array(image_seg)
            
            image = cv2.resize(image_seg, (224,224), interpolation=cv2.INTER_AREA)
            image = np.expand_dims(image, axis=-1) / 255.0
            image_batch = np.expand_dims(image, axis=0) 
            predicted_mask = segmentation_model.predict(image_batch)
            _, buffer = cv2.imencode('.png', predicted_mask.squeeze() *255)  # Encode the image to buffer
            encoded_image = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string for HTML display
            return encoded_image
    except Exception as e:
        print("Failed to load or process .mat file:", str(e))
        raise




def load_mat_and_preprocess(file_path):
    """
    Load a .mat file, extract the image for prediction, and return both the image data
    and its Base64 encoded string for display.
    """
    try:
        with h5py.File(file_path, 'r') as file:
            image_data = find_image_in_hdf5(file)  # Extract the image data
            if image_data is None:
                print("No suitable image dataset found in the .mat file.")
                raise ValueError("No suitable image dataset found in the .mat file.")
            
            # Ensure image_data is a numpy array and in the correct format
            if not isinstance(image_data, np.ndarray):
                image_data = np.array(image_data)
            
            # Check and adjust data type for encoding
            if image_data.dtype != np.uint8:
                image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)  # Normalize and convert

            # Encode the image for display
            _, buffer = cv2.imencode('.png', image_data)  # Encode the image to buffer
            encoded_image = base64.b64encode(buffer).decode('utf-8')  # Convert to base64 string for HTML display

            return image_data, encoded_image
    except Exception as e:
        print("Failed to load or process .mat file:", str(e))
        raise


def preprocess_image_array(image_array):
    """
    Assuming image_array is a NumPy array from the uploaded image file.
    Resize and preprocess.
    """
    # Check if the image data is valid
    if image_array is None or image_array.size == 0:
        raise ValueError("Empty or invalid image data received.")

    print("Image data received with shape:", image_array.shape)  # Debug output

    # Ensure the image is in the correct format (e.g., grayscale for single-channel operations)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # Assuming RGB image
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        print("Converted RGB to Grayscale.")

    # Only resize if the dimensions are not already the desired size
    if image_array.shape[0] != 224 or image_array.shape[1] != 224:
        try:
            image_resized = cv2.resize(image_array, (224, 224))
            print("Resized image to (224, 224).")
        except Exception as e:
            print(f"Failed to resize the image: {e}")
            raise ValueError(f"Failed to resize the image: {e}")
    else:
        image_resized = image_array
        print("Image already at the desired resolution.")

    try:
        image_stripped = skull_strip(image_resized)
        image_reduced_noise = noise_reduction(image_stripped)
        image_enhanced_contrast = contrast_enhancement(image_reduced_noise)
        image_normalized = normalize_image(image_enhanced_contrast)

        final_image = image_normalized.reshape((1, 224, 224, 1)).astype('float32')
        print("Image processed and normalized.")
        return final_image
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise


def skull_strip(image, fill_value=0):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    binary = morphology.remove_small_objects(binary, min_size=64)
    binary = morphology.binary_closing(binary, morphology.disk(3))
    binary = binary_fill_holes(binary)
    return np.where(binary, image, fill_value)

def noise_reduction(image, kernel_size=3):
    if image.dtype == np.float64:
        image = np.uint8(image * 255)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred.astype(np.float64) / 255.0

def contrast_enhancement(image):
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return image

def normalize_image(image):
    min_val, max_val = np.min(image), np.max(image)
    image = (image - min_val) / (max_val - min_val)
    return image

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/send_otp', methods=['POST'])
def send_otp():
    email = request.form.get('email')
    otp = ''.join(random.choices(string.digits, k=6))
    otps[email] = otp
    session['email_for_otp'] = email
    msg = Message('Your OTP for Brain Tumor Analysis', sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = (
        "Welcome to DeepScan: Your Brain Tumor Analysis Advisor! 🧠✨\n\n"
        "Hello! We're thrilled to have you with us. Your privacy and security are our top priorities.\n\n"
        f"Here's your One-Time Password (OTP): {otp} - Please keep it confidential and do not share it with anyone.\n\n"
        "Didn't request this OTP? If this message reached you by mistake, please let us know. Your security is crucial to us, and you can safely disregard this message otherwise.\n\n"
        "Thank you for choosing DeepScan. We're here to support your journey towards better health."
    )
    mail.send(msg)
    flash('OTP has been sent to your email.')
    return jsonify({'success': True})

@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    email = session.get('email_for_otp')
    user_otp = request.form.get('otp')
    if otps.get(email) == user_otp:
        del otps[email]
        session['otp_verified'] = True
        return redirect(url_for('app_route'))
    else:
        flash('Invalid OTP. Please try again.')
        return jsonify({'success': False}), 400

@app.route('/app', methods=['GET', 'POST'])
def app_route():
    if not session.get('otp_verified'):
        return redirect(url_for('home'))
    if request.method == 'POST':
        file = request.files.get('mri_image')
        if file and file.filename:
            extension = file.filename.rsplit('.', 1)[1].lower()
            if extension in ['jpg', 'jpeg']:
                # Read the file into a buffer
                buffer = file.read()
                if buffer:
                    image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        flash('Failed to decode the image. Please ensure it is a valid JPG file.')
                        return redirect(url_for('app_route'))
                else:
                    flash('Uploaded file is empty.')
                    return redirect(url_for('app_route'))
            elif extension == 'mat':
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                file.save(temp_file.name)
                try:
                    image = load_mat_and_preprocess(temp_file.name)
                finally:
                    os.unlink(temp_file.name)
            else:
                flash('Unsupported file format. Please upload a JPG or MAT file.')
                return redirect(url_for('app_route'))
            
            preprocessed_image = preprocess_image_array(image)
            predictions = model.predict(preprocessed_image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            class_probabilities = predictions[0]
            send_result_email(session['email_for_otp'], predicted_class, class_probabilities)
            flash('Results sent to your email!')
        else:
            flash('Please upload an MRI image.')
    return render_template('info.html')

@app.route('/submit_details', methods=['POST'])
def submit_details():
    if not session.get('otp_verified'):
        flash("OTP verification required.")
        return redirect(url_for('home'))

    file = request.files.get('mri_image')
    if file and file.filename:
        extension = file.filename.rsplit('.', 1)[1].lower()
        if extension not in ['jpg', 'jpeg', 'png', 'mat']:
            flash('Unsupported file format. Please upload a JPG, PNG, or MAT file.')
            return redirect(url_for('app_route'))

        # Create a temporary directory to save the file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)

        try:
            if extension == 'mat':
                # Load .mat file and get both image data and encoded image
                image_data, encoded_image = load_mat_and_preprocess(temp_file_path)
                image = preprocess_image_array(image_data)
                image_seg = load_mat(temp_file_path)
                predictions = model.predict(image.reshape((1, 224, 224, 1)))
                predicted_class = np.argmax(predictions, axis=1)[0]
                class_probabilities = predictions[0].tolist()  # Convert to list for easier JSON handling
                if predicted_class != 0:
                    return render_template('ResultsPage.html', image_data=encoded_image, predicted_class=predicted_class, class_probabilities=class_probabilities,segmented_image=image_seg)           
                else:
                    black_image = np.zeros((224, 224, 3), dtype=np.uint8)
                    image_pil = Image.fromarray(black_image)
                    buffer = io.BytesIO()
                    image_pil.save(buffer, format="PNG")
                    black_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return render_template('ResultsPage.html', image_data=encoded_image, predicted_class=predicted_class, class_probabilities=class_probabilities,black_image_base64=black_image_base64)

            elif extension in ['jpg' ,'jpeg' ,'png']:
                # Read the file once after saving it to a temp location
               with open(temp_file_path, "rb") as image_file:
                buffer = image_file.read()
                if buffer:
                    # Decode image for processing
                    image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if image is None or image.size == 0:
                        raise ValueError("Failed to decode the image or the image is empty.")
                    # Encode the image for display
                    encoded_image = base64.b64encode(buffer).decode('utf-8')
                    
                    # Process the image if it's not a .mat file
                    image = preprocess_image_array(image)
                    predictions = model.predict(image.reshape((1, 224, 224, 1)))
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    class_probabilities = predictions[0].tolist()  # Convert to list for easier JSON handling
                    black_image = np.zeros((224, 224, 3), dtype=np.uint8)
                    image_pil = Image.fromarray(black_image)
                    buffer = io.BytesIO()
                    image_pil.save(buffer, format="PNG")
                    black_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    if predicted_class != 0:
                        return render_template('ResultsPage.html', image_data=encoded_image, predicted_class=predicted_class, class_probabilities=class_probabilities,segmented_image=black_image_base64)           
                    else:
                        return render_template('ResultsPage.html', image_data=encoded_image, predicted_class=predicted_class, class_probabilities=class_probabilities,black_image_base64=black_image_base64)


            else:
                raise ValueError("Uploaded file is empty.")
        
        except Exception as e:
            flash(str(e))
            return redirect(url_for('app_route'))
        finally:
            shutil.rmtree(temp_dir)  # Clean up the temporary directory
    else:
        flash('No MRI image uploaded. Please try again.')
        return redirect(url_for('app_route'))

def send_result_email(email, predicted_class, class_probabilities):
    classes = ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary Tumor']
    body = f"Brain Tumor Analysis Results\n\nPredicted Condition: {classes[predicted_class]}\n\n"
    for i, class_name in enumerate(classes):
        body += f"{class_name}: {class_probabilities[i]:.4f}\n"
    msg = Message('Your Brain Tumor Analysis Result', sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = body
    mail.send(msg)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
