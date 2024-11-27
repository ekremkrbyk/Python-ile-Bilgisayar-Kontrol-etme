import cv2
import time
import numpy as np
import el_izleme as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

camera = cv2.VideoCapture(0)
aygitlar = AudioUtilities.GetSpeakers()

interface = aygitlar.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    
volume = interface.QueryInterface(IAudioEndpointVolume)

volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

Wcam ,Hcam = 640,480

camera.set(3,Wcam)
camera.set(4,Hcam)
pTime = 0 #Önceki zaman

dedektor = htm.handDetector(detectionCon=0.5)

sesBar = 400
sesYuzde = 0
while True:
    #Camera dan değer okuma
    success , kare = camera.read()  #Okundu mu,her bir kare
    kare = dedektor.findHands(kare) #Önceki projeden elleri tanıyıp eklem noktalarından belirleyen fonksiyon.
    lmList = dedektor.findPosition(kare,False)
    mesafe = 30    
    if len(lmList) >=8:
        # print(lmList[4],lmList[8]) #Baş ve işaret parmağının id ve kordinat bilgilerini yazdır.
        x1 ,y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx,cy = (x1 + x2) //2, (y1 + y2) //2 #2 ye böl ve integer değer ver. #Merkez dairenin kordinatları

        cv2.circle(kare,(x1,y1),5,(255,0,0),cv2.FILLED) #Baş parmağın mavi noktası
        cv2.circle(kare,(x2,y2),5,(255,0,0),cv2.FILLED) #İşaret parmağının mavi noktası
        cv2.line(kare,(x1,y1),(x2,y2),(255,0,0),5) #Arasındaki çizgi
        cv2.circle(kare,(cx,cy),5,(0,0,255),cv2.FILLED) #Merkez daire
        lenth = math.hypot(x2 - x1,y2 - y1)
        print(lenth)

        vol = np.interp(lenth*2,[10,160],[minVol,maxVol])
        sesBar = np.interp(lenth *2,[10,160],[400,150])
        sesYuzde = np.interp(lenth *2,[10,160],[0,100])

        print(f"Mesafe: {int(lenth)}") #Baş parmak ile işaret parmağı arasındaki uzaklık mesafe.
        
        volume.SetMasterVolumeLevel(vol, None)
        
        if lenth < 10:  
            cv2.circle(kare,(cx,cy),5,(0,155,0),3)
            print("Parmaklar Birleştirildi.")
            mesafe = lenth
    cv2.rectangle(kare,(150,50),(400,85),(155,0,0),5)
    cv2.rectangle(kare,(int(sesBar),50),(400,85),(155,0,0),cv2.FILLED)
    
    
    cTime = time.time() #Güncel zaman
    fps = 1 / (cTime - pTime)
    pTime = cTime
     #Görüntü ters gelmesin diye x ekseninde çevirdim.
    kare = cv2.flip(kare,1)
    cv2.putText(kare,f"FPS: {int(fps)}",(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,155,0),2)
    if mesafe < 10:
        cv2.putText(kare,"Parmaklar Birlestirildi.",(200,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0),3)
    cv2.imshow("Result",kare)
    c = cv2.waitKey(1) 
    if c == 27:
        break
camera.release()
cv2.destroyAllWindows()

     



