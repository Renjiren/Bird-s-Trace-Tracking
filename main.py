import preprocess

if __name__ == "__main__":
    preprocess.preprocess_frame(
        val_path="val",
        output_path="val_pre/gaussian",
        mode="gaussian"
    )