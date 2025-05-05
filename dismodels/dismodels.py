from ultralytics import YOLO
import cv2 as cv
import os
from PIL import Image
from kb import ROOT
from PIL import Image, ImageDraw, ImageFont

# img = f"{ROOT}/images/dis777.jpg"
def detection(img):
    link = f"{ROOT}/dismodels/best.pt"
    dis = []
    model = YOLO(f"{link}")
    results = model(f"{img}")
    res = model.predict(source=img, save=True)
    #model.predict(source=f"{img}", project=f"{ROOT}/static/dis/", save=True)
    a = list(map(lambda num: num.boxes.cls.tolist(), results))[0]
    b = list(map(lambda num: num.boxes.xyxy.tolist(), results))[0]
    n = results[0].names
    print(n)
    for each in a:
        nn = int(each)
        name = n[nn]
        dis.append(nn)
        print(name)
    return dis

def compil(cord, adr, n):
    sp = []
    img = Image.open(adr)
    for box in cord:
        sr = []
        x1, y1, x2, y2 = box
        # изображение каждой отдельной ягоды

        """далее для отрисовки"""
        myFont = ImageFont.truetype('arial.ttf', 25)
        ImageDraw.Draw(img).text((x1, y1), n, font=myFont, fill=(100, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=2)
        img.save(f"{ROOT}/static/dis/dis777.jpg", quality=100)


# detection(link, img)


