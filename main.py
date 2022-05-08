from mtcnn.mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

filename = "/home/appuser/venv/lib/python3.7/site-packages/keras_vggface/models.py"
text = open(filename).read()
open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

choice = st.selectbox("Select Option", [
    "Face Detection",
    "Face Detection 2",
    "Face Verification"
])

def extract_face(file):
    pixels = np.asarray(file)
    plt.axis("off")
    plt.imshow(pixels)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    return face_array

def main():
    fig = plt.figure()
    if choice == "Face Detection":
        uploaded_file = st.file_uploader("Choose File", type=['jpg','png'])
        if uploaded_file is not None:
            data = np.asarray(Image.open(uploaded_file))
            plt.axis("off")
            plt.imshow(data)
            ax = plt.gca()

            detector = MTCNN()
            faces = detector.detect_faces(data)
            for face in faces:
                x,y,w,h = face['box']
                rect = Rectangle((x,y), w,h, fill=False, color='maroon')
                ax.add_patch(rect)
                for _,value in face['keypoints'].items():
                    dot = Circle(value, radius=2, color='maroon')
                    ax.add_patch(dot)
                st.pyplot(fig)
    elif choice == "Face Detection 2":
        uploaded_file = st.file_uploader("Choose File", type=["jpg", "png"])
        if uploaded_file is not None:
            column1, column2 = st.columns(2)
            image = Image.open(uploaded_file)
            with column1:
                size = 450, 450
                resized_image = image.thumbnail(size)
                image.save("thumb.png")
                st.image("thumb.png")
            pixels = np.asarray(image)
            plt.axis("off")
            plt.imshow(pixels)
            detector = MTCNN()
            results = detector.detect_faces(pixels)
            x1, y1, width, height = results[0]["box"]
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize((224, 224))
            face_array = np.asarray(image)
            with column2:
                plt.imshow(face_array)
                st.pyplot(fig)

    elif choice == "Face Verification":
        column1, column2 = st.columns(2)

        with column1:
            image1 = st.file_uploader("Choose File", type=["jpg", "png"])

        with column2:
            image2 = st.file_uploader("Select File", type=["jpg", "png"])
        if (image1 is not None) & (image2 is not None):
            col1, col2 = st.columns(2)
            image1 = Image.open(image1)
            image2 = Image.open(image2)
            with col1:
                st.image(image1)
            with col2:
                st.image(image2)

            filenames = [image1, image2]

            faces = [extract_face(f) for f in filenames]
            samples = np.asarray(faces, "float32")
            samples = preprocess_input(samples, version=2)
            model = VGGFace(model="resnet50", include_top=False, input_shape=(224, 224, 3),
                            pooling="avg")
            # perform prediction
            embeddings = model.predict(samples)
            thresh = 0.5
            score = cosine(embeddings[0], embeddings[1])
            if score <= thresh:
                st.success(" >face is a match (%.3f <= %.3f) " % (score, thresh))
            else:
                st.error(" >face is NOT a match (%.3f > %.3f)" % (score, thresh))


if __name__ == "__main__":
    main()