from PIL import Image
import numpy as np
import tkinter
import pytesseract
import cv2
import os

"""
função especifica para encontrar os contornos na imagem que mas se 
assemelhe com os contornos retangulo de uma placa um veiculo
"""


def desenhaContornos(contornos, imagem):
    for c in contornos:

        perimetro = cv2.arcLength(c, True)
        if perimetro > 80:

            approx = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            # verifica se é um quadrado ou retangulo de acordo com a qtd de vertices
            if len(approx) == 4:
                # cv2.drawContours(imagem, [c], -1, (0, 255, 0), 1)
                (x, y, a, l) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + a, y + l), (0, 255, 0), 2)
                roi = imagem[y:y + l, x:x + a]
                cv2.imwrite(IMAGE_DIR + '/roi.jpg', roi)

    return imagem


def reconhecimentoOCR(path_img):
    entrada = cv2.imread(path_img + ".jpg")
    img = cv2.resize(entrada, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("Limiar", img)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(path_img + "-ocr.jpg", img)
    imagem = Image.open(path_img + "-ocr.jpg")
    saida = pytesseract.image_to_string(imagem, lang='eng')
    print(saida)
    texto = removerChars(saida)
    janela = tkinter.Tk()
    tkinter.Label(janela, text=texto, font=("Helvetica", 50)).pack()
    janela.mainloop()
    cv2.destroyAllWindows()


def removerChars(self, text):
    str = "!@#%¨&*()_+:;><^^}{`?|~¬\/=,.'ºª»‘"
    for x in str:
        text = text.replace(x, '')
    return text


ROOT_DIR = os.path.dirname(os.path.abspath(''))
IMAGE_DIR = os.path.join(ROOT_DIR, 'src/images/')
file = 'car_plate.jpeg'
img = cv2.imread(IMAGE_DIR + file)
img = cv2.resize(img, (720, 480))

"""
    define limites na imagem a qual sera encontrada os contornos
"""
cv2.line(img, (0, 350), (860, 350), (0, 0, 255), 1)
# limite vertical 1
cv2.line(img, (220, 0), (220, 480), (0, 0, 255), 1)
# limite vertical 2
cv2.line(img, (500, 0), (500, 480), (0, 0, 255), 1)

cv2.imshow('SAIDA', img)

res = img[350:, 220:500]

"""
    pre procesamento imagem, tem o objetivo de diminuir os ruidos da imagem
"""
img_result = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# limiarização
ret, img_result = cv2.threshold(img_result, 90, 255, cv2.THRESH_BINARY)

# desfoque
img_result = cv2.GaussianBlur(img_result, (5, 5), 0)

# lista os contornos
contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

desenhaContornos(contornos, res)

cv2.imshow('RES', res)

"""
    não foi possivel aplicar o reconhecimento dos numeros da placa 
    porque o algoritimo de reconhecimento de contornos não identifica
    corretamente o local da placa do veiculo.
"""
# reconhecimentoOCR(IMAGE_DIR + '/roi')
cv2.waitKey(0)
cv2.destroyAllWindows()
