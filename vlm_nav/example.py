import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from vlm_nav import VLMNavAgent
from vlm_nav.vlm_seg.semantic_label_processor import generate_vocab_image

parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument("image_path", type=str, help="The path to the image file")
parser.add_argument(
    "-i", "--image_path2", type=str, help="The path to the second image file (optional)", required=False
)
parser.add_argument(
    "-c", "--config_path", type=str, default="config/prompts.yaml", help="The path to the configuration file"
)
parser.add_argument(
    "-t", "--task", type=str, default="nature_hiking", help="The prompt setup key in the config file"
)

parser.add_argument("-s", "--seg_model", type=str, default="SAN", help="The segmentation model to use. SAN or api")
parser.add_argument("-l", "--labels", type=str, required=False, help="The json label input as string")
parser.add_argument("-d", "--debug_visualizations", action="store_true", help="Enable debug visualizations")

parser.add_argument(
    "--vlm_model", type=str, default="gpt-6.1-sol", help="OpenAI (gpt-*) or Gemini (gemini-*) model ID"
)
parser.add_argument("--half_precision", action="store_true", help="Use half precision for the VLM model")
parser.add_argument("--save_images", action="store_true", help="Save images after processing")

args = parser.parse_args()

image_path = args.image_path
vlm_agent = VLMNavAgent(
    args.config_path,
    args.task,
    segmentation_model=args.seg_model,
    vlm_model=args.vlm_model,
    half_precision=args.half_precision,
)

print("Loading the image...", image_path)
image = Image.open(image_path)
print("Image size:", image.size)
# Show the image

if args.image_path2:
    print("Loading the second image...", args.image_path2)
    image2 = Image.open(args.image_path2)
    images = [image, image2]
else:
    images = [image]

max_size = 640, 640
for img in images:
    img.thumbnail(max_size)

for img in images:
    print("Using the image...", img.size)


if args.labels:
    import json

    labels = json.loads(args.labels)
    print("Using the provided labels:", labels)
else:
    print("Asking VLM to analyze the image...")
    labels, response = vlm_agent.analyze(images, return_response=True)
    print("VLM response:", response)
    print("Estimated labels:")
    for label, k in labels.items():
        print(f"- {label}: {k}")


print("images", images)
for image in images:
    print("labels", labels)
    seg_return = vlm_agent.segment(image, labels, output_debug_info=True)
    if seg_return is not None:
        seg_image, seg_labels, seg_debug_info = seg_return

        if args.debug_visualizations:
            img = generate_vocab_image(seg_debug_info["normalized"], seg_debug_info["vocab"])
            plt.show()

        num_labels = len(seg_labels)
        fig, axs = plt.subplots(1, num_labels + 1, figsize=(15, 5))  # Adjusted for the additional input image
        # Display the input image in the first subplot
        axs[0].set_title("Input Image")
        axs[0].imshow(image)
        axs[0].axis("off")  # Hide the axis for the input image

        for i, label in enumerate(seg_labels, start=1):  # Start enumeration at 1 to leave space for the input image
            seg_image_np = seg_image[i - 1].to("cpu").numpy()  # Adjust index for seg_image
            axs[i].set_title(label)
            im = axs[i].imshow(seg_image_np, cmap="jet", interpolation="nearest", vmin=0, vmax=1)
            fig.colorbar(im, ax=axs[i])
        plt.show()

        if args.save_images:
            os.makedirs("result", exist_ok=True)
            image_name = os.path.splitext(os.path.basename(image.filename))[0]
            specific_image_dir = f"result/{image_name}"
            os.makedirs(specific_image_dir, exist_ok=True)
            print(f"Saving images to 'result/{image_name}_input.png'...")
            image.save(f"{specific_image_dir}/input.png")
            for i, label in enumerate(seg_labels):
                seg_image_np = seg_image[i].to("cpu").numpy()
                plt.imsave(f"{specific_image_dir}/{label}.png", seg_image_np, cmap="jet", vmin=0, vmax=1)
                seg_image_gray = np.uint8(seg_image_np * 255)  # Convert to 8-bit grayscale
                Image.fromarray(seg_image_gray).save(f"{specific_image_dir}/{label}_gray.png")
