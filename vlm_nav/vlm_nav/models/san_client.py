import requests
from PIL import Image
import numpy as np
import io
import base64
import torch
from matplotlib import pyplot as plt
from typing import Union, Optional
import time


url = "http://localhost:7860/predict"


def san_predict(image: Image, vocabulary, augment_vocabulary: str = "", return_all_categories: bool = False):
    # Save the PIL image to a byte buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  # You can change the format to PNG if preferred
    buffer.seek(0)

    # Prepare the data for POST request
    vocab_string = ",".join(vocabulary)
    files = {"image": buffer}
    data = {"vocab": vocab_string, "return_all_categories": return_all_categories}  # Replace with your vocabulary list
    s = time.time()
    response = requests.post(url, files=files, data=data)
    response = response.json()

    vocabulary = response["vocabulary"]

    # Convert the response's "result" to a numpy array
    base64_encoded = response["result"]
    tensor_bytes = base64.b64decode(base64_encoded)
    numpy_array = np.frombuffer(tensor_bytes, dtype=np.float32)
    numpy_array = numpy_array.reshape(response["shape"])

    return {"result": numpy_array, "vocabulary": vocabulary}


def san_predict_from_file(
    image_path: str,
    vocabulary,
    visualize: bool = False,
    return_all_categories: bool = False,
    weights: dict = None,
    save_path: str = None,
):
    image = Image.open(image_path)
    max_size = (640, 640)
    image.thumbnail(max_size, Image.ANTIALIAS)
    results = san_predict(image, vocabulary, return_all_categories)
    arrays = results["result"]
    vocabulary = results["vocabulary"]
    if visualize:
        visualize_heatmaps(image, arrays, vocabulary, weights, save_path)
    return arrays, vocabulary


def visualize_heatmaps(image, arrays, vocabulary, weights=None, save_path: Optional[str] = None):
    # Visualize with the argmax of the categories
    fig, axs = plt.subplots(1, 2)
    # Visualize the original image
    axs[0].imshow(image)
    axs[0].set_title("Original Image")

    arrays_copy = arrays.copy()
    if weights is not None:
        for k, v in weights.items():
            idx = vocabulary.index(k)
            arrays_copy[idx] *= v
            print("k", k, "v", v, "idx", idx)

    # Visualize the argmax of categories
    max_index = np.argmax(arrays_copy, axis=0)
    im = axs[1].imshow(image, alpha=0.5)
    im = axs[1].imshow(max_index, cmap=plt.cm.get_cmap("tab10", len(vocabulary)), interpolation="nearest", alpha=0.7)
    fig.colorbar(
        im, ax=axs[1], ticks=range(len(vocabulary)), format=plt.FuncFormatter(lambda val, loc: vocabulary[val])
    )
    axs[1].set_title("Segmentations")

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    for i, vocab in enumerate(vocabulary):
        plt.figure(figsize=(10, 10))
        plt.imshow(arrays[i], cmap="hot", interpolation="nearest", vmin=0, vmax=1)
        plt.title(vocab)
        if save_path is not None:
            plt.savefig(f"{save_path}_{vocab}_heatmap.png")
        else:
            plt.show()
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict semantic labels for an image.")
    parser.add_argument("--image_path", default="", type=str, help="Path to the image file.")
    parser.add_argument("--image_dir", default="", type=str, help="Path to the image file.")
    parser.add_argument("-v", "--vocabulary", type=str, nargs="+", help="List of vocabularies.")
    parser.add_argument("--weights", type=float, nargs="+", help="List of weights.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the heatmaps.")
    parser.add_argument("--return_all_categories", action="store_true", help="Use all categories.")

    args = parser.parse_args()

    image_path = args.image_path
    vocabulary = args.vocabulary
    if args.weights is not None:
        weights = {vocabulary[i]: args.weights[i] for i in range(len(vocabulary))}
    else:
        weights = None
    visualize = args.visualize
    print("vocab", vocabulary)

    if args.image_dir != "":
        import os

        save_dir = os.path.join(args.image_dir, "segmentations")
        os.makedirs(save_dir, exist_ok=True)
        for filename in os.listdir(args.image_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(args.image_dir, filename)
                save_path = os.path.join(save_dir, filename.split(".")[0] + "_segmentation.png")
                arrays, vocabulary = san_predict_from_file(
                    image_path, vocabulary, visualize, args.return_all_categories, weights, save_path
                )
                print("Semantic labels for", filename, ":", vocabulary)
    else:
        arrays, vocabulary = san_predict_from_file(image_path, vocabulary, visualize, args.return_all_categories)
        print("Semantic labels:", vocabulary)


if __name__ == "__main__":
    main()
