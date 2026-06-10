from PIL import Image

img = Image.open("C:/PBRPacker/PBRPacker.ico")
icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save("C:/PBRPacker/app_icon.ico", format="ICO", sizes=icon_sizes)
