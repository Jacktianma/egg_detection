import joblib
scaler = joblib.load("scaler.pkl")

print("MEAN =", scaler.mean_.tolist())
print("STD =", scaler.scale_.tolist())
