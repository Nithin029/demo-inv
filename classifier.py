import numpy as np
import onnxruntime


class Classifier:
    def __init__(self, onnx_fp: str) -> None:
        try:
            self.classifier = onnxruntime.InferenceSession(path_or_bytes=onnx_fp)
        except Exception as e:
            print(e)

    def preprocess(self, img):
        """
        img : PIL Image object of shape (B,HxW,C)
        """
        img = img.resize((192,192))
        np_image = np.asarray(img) / 255
        return np_image.astype(np.float32)

    def classify(self, imgs):
        # preprocess
        processed_imgs = []
        for img in imgs:
            pi = self.preprocess(img)
            processed_imgs.append(pi)

        batch = np.array(processed_imgs)
        onnx_input = {"images": batch}
        prediction = self.classifier.run(None, onnx_input)

        return (prediction[0] > 0.5).astype(np.int8).flatten().tolist()
