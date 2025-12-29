import cv2
import random

# ----------------------
# Step 1: Movie Database
# ----------------------
movies = {
    "happy": ["Paddington 2", "Singin in the Rain", "Zootopia", "The Intouchables", "School of Rock"],
    "sad": ["Schindler's List", "Manchester by the Sea", "Marley & Me", "Room", "Blue Valentine"],
    "surprise": ["Inception", "Shutter Island", "The Sixth Sense", "Gone Girl", "Arrival"],
    "fear": ["Get Out", "Hereditary", "A Quiet Place", "The Conjuring", "Psycho"],
    "angry": ["Kill Bill", "John Wick", "Gladiator", "Django Unchained", "Oldboy"],
    "disgust": ["Trainspotting", "Saw", "The Fly", "Requiem for a Dream", "Alien"],
    "contempt": ["Mean Girls", "The Devil Wears Prada", "Gone Girl", "Fight Club", "American Psycho"],
    "neutral": ["Forrest Gump", "Cast Away", "The Shawshank Redemption"]
}

# ----------------------
# Step 2: Load Haar Cascades
# ----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# ----------------------
# Step 3: Open webcam
# ----------------------
cap = cv2.VideoCapture(0)
print("Press 'q' to quit and get movie recommendations...")

detected_emotion = "neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]

        # Draw green rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ----------------------
        # Step 4: Basic Emotion Detection
        # ----------------------
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

        if len(smiles) > 0:
            detected_emotion = "happy"
        elif len(eyes) > 2:
            detected_emotion = "surprise"
        else:
            detected_emotion = random.choice(["sad", "angry", "fear", "disgust", "contempt", "neutral"])

        # Show detected emotion *above the rectangle*
        cv2.putText(frame, f"{detected_emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # green text to match rectangle

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------
# Step 5: Recommend movies
# ----------------------
print(f"\nDetected Emotion: {detected_emotion}")
recommendations = random.sample(movies[detected_emotion], min(3, len(movies[detected_emotion])))
print("🎬 Recommended Movies for you:")
for m in recommendations:
    print("-", m)