from core.ml.predictor import predict

if __name__ == "__main__":

    path = input("CSV Path: ").strip()

    predict(path)