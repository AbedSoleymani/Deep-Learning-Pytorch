import imageio
from PIL import Image, ImageDraw, ImageFont
import os

def save_gif(step):
    
    image_directory = "./GenerativeAI/DCGAN/logs/fake/"

    output_gif_path = "./GenerativeAI/DCGAN/generated_gif.gif"

    image_paths = []

    for i in range(step):
        image_paths.append(os.path.join(image_directory, f"{i}"))

    with imageio.get_writer(output_gif_path, duration=0.5) as gif_writer:
        for i, image_path in enumerate(image_paths):
            image = imageio.imread(f"{image_path}.png")

            step_number_text = f"Step: {i+1}"
            image = Image.fromarray(image)
            draw = ImageDraw.Draw(image)
            
            font = ImageFont.load_default()
            
            draw.text((10, 10), step_number_text, font=font, fill=(255, 0, 0))


            gif_writer.append_data(image)

    print(f"GIF created and saved at: {output_gif_path}")