import dlib
import cv2
import os
import glob
import sys

def main():
    #treinamento()
    #treinamento_pontos()
    #detecta_pl_IMG()
    #detecta_pl_com_pontos()
    detecta_pl_VID()

def treinamento():# algoritmo SVM (Support Vector Machine)
    op = dlib.simple_object_detector_training_options()#função para treino
    op.add_left_right_image_flips = True #ele muda a rotação da imagem diversas vezes para achar as placas
    op.C = 20 # punição de aprendizado (usado para minimizar o erro do algoritmo)
    dlib.train_simple_object_detector("treino_placas.xml", "detector_placas.svm", op)# pega o arquivo .xml para treinar e guardar o treino em um .svm

def detecta_pl_IMG():
    detectpl = dlib.simple_object_detector("detector_placas.svm")
    for imagem in glob.glob(os.path.join("TestesPlacas","*.jpg")):
        img = cv2.imread(imagem)
        placasDetectadas = detectpl(img, 2)
        for d in placasDetectadas:
            e,t,d,b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
            cv2.rectangle(img, (e,t), (d,b), (0,0,255), 2)
        cv2.imshow("Detecta placas", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def detecta_pl_VID():
    taxa_quadros = 30
    captura = cv2.VideoCapture("Video.mp4")
    contadorQuadros = 0
    detector = dlib.simple_object_detector("detector_placas.svm")
    while(True):
        conectado, frame = captura.read()
        contadorQuadros +=1
        if contadorQuadros % taxa_quadros == 0:
            Pldetectadas = detector(frame)
            for p in Pldetectadas:
                e, t, d, f = (int(p.left()), int(p.top()), int(p.right()), int(p.bottom()))
                cv2.rectangle(frame, (e,t), (d,f), (0,0,255), 2)
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xff == 27:#Se apertar "Esc" ele sai do while
                break
    captura.release()
    cv2.destroyAllWindows()
    sys.exit(0)

#parte em teste sobre preditor de forma (usado para aperfeiçoar o reconhecimento de placa)
'''    
def treinamento_pontos():
    op = dlib.shape_predictor_training_options()
    dlib.train_shape_predictor("treino_placas_pontos.xml","treino_placas_pontos.dat", op)
    '''

'''
def imprimirPontos(imagem, pontos):
    for p in pontos.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0))

def detecta_pl_com_pontos():
    detectorPl = dlib.simple_object_detector("detector_placas.svm")
    detectorPPl = dlib.shape_predictor("treino_placas_pontos.dat")
    for arquivo in glob.glob(os.path.join("TestesPlacas", "*.jpg")):
        imagem = cv2.imread(arquivo)
        PlacasDetectadas = detectorPl(imagem, 2)
        for placas in PlacasDetectadas:
            e, t, d, b = (int(placas.left()), int(placas.top()), int(placas.right()), int(placas.bottom()))
            cv2.rectangle(imagem, (e,t),(d,b),(0,0,255), 2)
            pontos = detectorPPl(imagem, placas)
            imprimirPontos(imagem, pontos)
        cv2.imshow("Detectar pontos", imagem)
        cv2.waitKey(0)
    cv2.destroyAllWindows
'''
main()
