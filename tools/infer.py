import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

def draw_detection(draw, box, label):
    color = "yellow"
    thickness = 2
    font = ImageFont.truetype("./work_dirs/arial.ttf", 10)
    
    xmin, ymin, xmax, ymax, score = box[0]

    ymin = int(ymin)
    xmin = int(xmin)
    ymax = int(ymax)
    xmax = int(xmax)
    label = "bus"

    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=thickness)

    draw.text((xmin, ymin), label, fill=color, font=font)

def main(model_path, input_path, output_path):
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

    dets = outputs[0]
    labels = outputs[1]

    draw = ImageDraw.Draw(img)

    for i in range(len(dets)):
        box = dets[i]
        label = labels[i]
        draw_detection(draw, box, label)

    img.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection using ONNX model")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to output image")
    args = parser.parse_args()

    main(args.model, args.input, args.output)    





