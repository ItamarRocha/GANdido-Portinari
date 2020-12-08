from PIL import Image
import os

folder = "gandido"

for file in os.listdir(folder):
    try:
        img = Image.open(folder + "/" + file)
        img = img.resize((256,256))

        img.save(f"gandido_resized/" + file)
    except:
        print(f"failed at {file}")
