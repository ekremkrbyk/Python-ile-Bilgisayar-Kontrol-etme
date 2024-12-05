import cv2
import mediapipe as mp
import time
import math
import numpy as np
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode #Görüntü modu statik görüntü mü yoksa gerçek zamanlı görüntü mü.
        self.maxHands = maxHands #Okunacak max el sayısı.
        self.detectionCon = detectionCon #Tespit doğruluğu güven ayarı
        self.trackCon = trackCon #Parmak uçlarının kodu. [0] [1] [2] gibi
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
        static_image_mode=bool(self.mode), #Görüntü modu eğer true gelirse her kar bağımsız işlenir. False gelirse önceki kareyi referans alarak işler.
        max_num_hands=int(self.maxHands),
        min_detection_confidence=float(self.detectionCon),
        min_tracking_confidence=float(self.trackCon))
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20] #Her parmağın uc noktasını belirtir.
 
    def findHands(self, img, draw=True): #El algılama
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Görüntüyü RGB formatına çevirir.
        self.results = self.hands.process(imgRGB)
     
        if self.results.multi_hand_landmarks: #Eğer bir el algılanırsa elin eklem noktalarını ve onların çizgilerini çizen kod
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, #img Her kareyi ,handLms ise parmakları temsil ediyor.
                    self.mpHands.HAND_CONNECTIONS) #Elin noktaları birleştiren çizgisi
        return img #El belirlenmiş ve etrafı çizilmiş bir şekilde görüntüyü dönüyor.
     
    def findPosition(self, img, handNo=0, draw=True):
        xmin = 0 #Elin etrafındaki kutu için min max noktaları
        xmax = 0
        ymin = 0
        ymax = 0
        xList = [] #Parmakların x ve y listesi  
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
     
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
     
        if draw: #Elin etrafındaki kareyi çizen kod
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
            (0, 255, 0), 2)
     
        return self.lmList #Parmakların listesi
     
    def fingersUp(self): #Parmakların açık mı kapalı mı olduğunu kontrol eder.
        fingers = []
                        #self.lmList elin anahtar noktaları   self.tipIds parmak uçları
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]: #Baş parmağı kontrol ediyor x eksenine göre.
            fingers.append(1)
        else:
            fingers.append(0)
     
        for id in range(1, 5): #Diğer parmaklar z eksenine göre kontrol ediyorum.
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
     
        return fingers  #Örneğin Çıktı Fingers: [1, 1, 0, 0, 0] bunun gibi liste şeklinde sırayla başparmaktan başlayarak serçe parmağa kadar gidiyor hangi parmağın açık olduğunu gösteriyor.
     
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3): #Parmaklar arasındaki mesafe p1 1. parmak p2 2. parmak draw çizim yapacak mı
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
     
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1) #Hipotenüs ile x ve y kordinatları arsındaki mesafeyi alıyor yani çizgi için gerekli kordinatlar
     
        return length, img, [x1, y1, x2, y2, cx, cy] #length iki nokta arasındaki mesafe
  
def main(): #Yukarıdakilerin hepsini çalıştıran ana fonksiyon
    pTime = 0 #Önceki zaman Önceki kare
    cTime = 0 #Sonraki zaman Sonraki kare
    camera = cv2.VideoCapture(0) #Dahili kamera ,harici kamera olsaydı 1 yazmam gerekirdi
    detector = handDetector()
    while True:
        success, img = camera.read()
        if success: #Kamera okundu mu
            img = cv2.flip(img,1) #Görüntüyü ters çevirdim
            img = detector.findHands(img)
            lmList = detector.findPosition(img) #lmList parmakların pozisonları.
            if len(lmList) != 0:
                print(lmList[4],lmList[8]) #Başparmağın pozisyonlarını yazdırır.
 
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
 
            cv2.putText(img, f"Fps: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
            (0,0,155), 3)

            cv2.imshow("Result", img)
            cv2.waitKey(1)
        else:
            print("Kamera okunamadı lütfen kameranızı kontrol ediniz.")
 
if __name__ == "__main__":
    main()