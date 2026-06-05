from tensorflow.keras.models import load_model

try:
    model = load_model("model.h5", compile=False)

    print("✅ model.h5 loaded successfully")
    model.summary()

except Exception as e:
    print("❌ model.h5 failed")
    print(e)