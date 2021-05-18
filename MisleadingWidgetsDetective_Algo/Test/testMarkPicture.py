from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
from PIL import ImageFont
ImageFont.truetype("NotoSansCJK-Regular.ttc", 30)
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
img_path = 'image1.png'
result = ocr.ocr(img_path, cls=True)
print(result)
img_path = 'image2.png'
result = ocr.ocr(img_path, cls=True)
print(result)
img_path = 'image3.png'
result = ocr.ocr(img_path, cls=True)
print(result)
img_path = 'image4.png'
result = ocr.ocr(img_path, cls=True)
print(result)
img_path = 'image5.png'
result = ocr.ocr(img_path, cls=True)
print(result)
'''
for line in result:
    print(line)
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores,font_path='/usr/share/fonts/truetype/NotoSansCJK-Regular.ttc')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
'''
