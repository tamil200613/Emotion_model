from keras.models import load_model

model = load_model("models/emotion_model.hdf5", compile=False)

print("Emotion model loaded successfully")
