# Future-Vision-Transport
Computer Vision 

How to use the app.py file:
1. Install the required libraries:
   - Open a terminal and run:
     ```
     pip install -r requirements.txt
     ```    
2. Run the application:
   - In the terminal, execute:
     ```
     python app.py
     ```
3. Access the application:
   - Open a web browser and go to:
     ```
     http://   
localhost:5000
     ```
4. Upload an image:
    - Use the provided interface to upload an image file.
5. View the results:
   - The application will process the image and display the results, including detected objects and their bounding boxes.
# Future Vision Transport
# Future Vision Transport is a computer vision application designed to detect and classify objects in images using deep learning models.    
# It provides a user-friendly interface for uploading images and viewing the results of object detection.
# The application uses Flask for the web interface and TensorFlow/Keras for the deep learning model.
# The application is structured to allow easy integration of different models and image processing techniques.
# The main components of the application include:
# - `app.py`: The main Flask application that handles image uploads and processing.
# - `model.py`: Contains the deep learning model for object detection.
# - `utils.py`: Contains utility functions for image processing and model inference.
# - `templates/`: Contains HTML templates for the web interface.
# - `static/`: Contains static files such as CSS and JavaScript for the web interface.
# The application is designed to be modular and extensible, allowing for easy updates and improvements.
# The application is built using Flask, a lightweight web framework for Python, and uses TensorFlow/Keras for deep learning.
# The application is designed to be modular and extensible, allowing for easy updates and improvements.
# The application is built using Flask, a lightweight web framework for Python, and uses TensorFlow/Keras for deep learning.

How to use the provided code:
1. Update CKPT_PATH and NUM_CLASSES as needed.
2. Install FastAPI and Uvicorn:
```bash
pip install fastapi uvicorn
```
3. Start the server:
```bash
uvicorn main:app --reload
```
4. POST an image to /segment/ and receive the overlayed image.