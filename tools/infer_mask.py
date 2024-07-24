import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

def draw_masks_pil(image, masks, labels, colors, alpha=0.3):
    """
    Annotates an image with segmentation masks, labels, and optional alpha blending.

    This function draws segmentation masks on the provided image using the given mask arrays, 
    colors, labels, and alpha values for transparency.

    Parameters:
    image (PIL.Image): The input image on which annotations will be drawn.
    masks (numpy.ndarray): A 3D numpy array of shape (n_masks, height, width) representing segmentation masks.
    labels (list of str): A list of labels corresponding to each segmentation mask.
    colors (list of tuples): A list of RGB tuples for each segmentation mask and its corresponding label.
    alpha (float, optional): The alpha value for mask transparency. Defaults to 0.3.

    Returns:
    annotated_image (PIL.Image): The image annotated with segmentation masks and labels.
    """
    annotated_image = image.copy()
    annotated_image = annotated_image.convert('RGBA')
    draw = ImageDraw.Draw(annotated_image)
    for i in range(len(labels)):
        mask = masks[i]
        mask_color = [*colors[i], int(alpha * 255)]
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgb_mask[mask > 0.5] = mask_color
        mask_img = Image.fromarray(rgb_mask, mode='RGBA')
        annotated_image.paste(mask_img, (0, 0), mask=mask_img)
    return annotated_image

def draw_bboxes_pil(image, boxes, labels, colors, font=None, width=2, font_size=18, probs=None):
    """
    Annotates an image with bounding boxes, labels, and optional probability scores.

    This function draws bounding boxes on the provided image using the given box coordinates, 
    colors, and labels. If probabilities are provided, they will be added to the labels.

    Parameters:
    image (PIL.Image): The input image on which annotations will be drawn.
    boxes (numpy.ndarray): An array of bounding box coordinates where each box has shape (1, 2).
    labels (list of str): A list of labels corresponding to each bounding box.
    colors (list of str): A list of colors for each bounding box and its corresponding label.
    font (PIL.ImageFont or None): The font object to use for drawing labels.
    width (int, optional): Width of the bounding box lines. Defaults to 2.
    font_size (int, optional): Size of the font for the labels. Defaults to 18.
    probs (list of float, optional): A list of probability scores corresponding to each label. Defaults to None.

    Returns:
    annotated_image (PIL.Image): The image annotated with bounding boxes, labels, and optional probability scores.
    """
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Load font
    if font is None:
        font = ImageFont.load_default()

    # Iterate over each bounding box and its label
    for box, label, color in zip(boxes, labels, colors):
        # Extract box coordinates
        x_min, y_min = box

        # Draw the bounding box
        shape = (x_min, y_min, x_min + 100, y_min + 100)  # Adjust the box size as needed
        draw.rectangle(shape, outline=color, width=width)

        # Draw label
        draw.text((x_min, y_min), str(label), fill=color, font=font)  # Ensure label is converted to string

    return annotated_image

def main(model_path, input_path, output_path):
    # Load the ONNX model
    sess = rt.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]

    # Inference session
    img = Image.open(input_path)
    img_data = np.array(img)
    img_data = img_data.astype(np.float32) / 255.0 
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    outputs = sess.run(output_names, {input_name: img_data})

    masks = outputs[0]
    boxes = outputs[1]
    labels = outputs[2]

    # Define labels and colors
    label_names = ["bus"]
    label_colors = [(0, 0, 255)]

    # Draw masks
    annotated_img = draw_masks_pil(img, masks, label_names, label_colors)
    
    # Load font
    font = ImageFont.truetype("./work_dirs/arial.ttf", 10)

    # Draw bounding boxes
    annotated_img = draw_bboxes_pil(annotated_img, boxes, labels, label_colors, font=font)

    # Convert image mode to RGB before saving as JPEG
    if annotated_img.mode != 'RGB':
        annotated_img = annotated_img.convert('RGB')

    # Save annotated image
    annotated_img.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Mask Segmentation using ONNX model")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    args = parser.parse_args()

    main(args.model, args.input, args.output)