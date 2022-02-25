import numpy
from function import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
from numpy import loadtxt

f=open("fold1.txt","r")
readLine=f.readlines() #ใช้เเค่ตอน read

#f1=open("buycomL2left.txt","w") เอาไว้เขียนลงไฟล์
#if ((X[i].count('<=30')==1)): ตอนเช็ค

w7 = ([   0.245,        4.201,       -0.103,       0.569,       0.006,       0.196,       0.614])
w8 = ([   0.103,       -4.040,        0.054,      -0.149,      -0.823,      -0.017,      -0.140])
w9 = ([   0.082,       -5.061,       -0.273,       0.626,       0.688,       0.255,       0.616])
w10=([  10.831,    -8.188,    -8.714,    -8.714])
w11=([  -1.735,    -1.058,     1.074,     1.065])
w12=([  -5.994,     4.814,     0.718,     0.718])

l = -0.9 # learning rate
epoch = 0
err10 = 1000
err11 = 1000
err12 = 1000

hitCount = 0
missCount = 0
for i in range(1,101) :
        separate = readLine[i].split(sep="\t", maxsplit=7)
        #print("%s %s %s %s %s %s %s"%(separate[0],separate[1],separate[2],separate[3],separate[4],separate[5],separate[6]))
        
        X  = ([int(separate[0]),int(separate[1]),int(separate[2]),int(separate[3]),int(separate[4]),int(separate[5]),int(separate[6])])
        
        d10 = 11 #desire output
        d11 = 11 #desire output
        d12 = 11 #desire output
        if( X[6] == 2):
            d10 = 1 #desire output
            d11 = 0 #desire output
            d12 = 0 #desire output
        elif( X[6] == 1):
            d10 = 0 #desire output
            d11 = 1 #desire output
            d12 = 0 #desire output
        elif( X[6] == 0):
            d10 = 0 #desire output
            d11 = 0 #desire output
            d12 = 1 #desire output
        
        
        #forward pass
        o7=Nout(X,w7)
        y7=sigmoid(o7)
        #print("\nOutput from node 7 is: %8.3f, Y from node 7 is: %8.3f" % (o7,y7))
        o8=Nout(X,w8)
        y8=sigmoid(o8)
        #print("\nOutput from node 8 is: %8.3f, Y from node 8 is: %8.3f" % (o8,y8))
        o9=Nout(X,w9)
        y9=sigmoid(o9)
        #print("\nOutput from node 9 is: %8.3f, Y from node 9 is: %8.3f" % (o9,y9))
        x10=([y7,y8,y9,d10])
        x11=([y7,y8,y9,d11])
        x12=([y7,y8,y9,d12])
        
        o10=Nout(x10,w10)
        o11=Nout(x11,w11)
        o12=Nout(x12,w12)
        y10=sigmoid(o10)
        y11=sigmoid(o11)
        y12=sigmoid(o12)
        #print("\nOutput from node 10 is: %8.3f, Y from node 10 is: %8.3f" % (o10,y10))
        #print("\nOutput from node 11 is: %8.3f, Y from node 11 is: %8.3f" % (o11,y11))
        #print("\nOutput from node 12 is: %8.3f, Y from node 12 is: %8.3f" % (o12,y12))
        predict = 9
        valide = "default"
        
        if((y10>0.5)and(y11<0.5)and(y12<0.5)): 
            predict = 2
        elif((y10<0.5)and(y11>0.5)and(y12<0.5)): 
            predict = 1
        elif((y10<0.5)and(y11<0.5)and(y12>0.5)): 
            predict = 0
        
        if(predict== X[6]):
            valid = "H"
            hitCount = hitCount + 1
        else:
            valid = "-"
            missCount = missCount + 1
        print("Row %d       Real class = %d       Predict class = %d      valid = %s"%(i,X[6],predict,valid))
print("Hit count = %d from %d\n"%(hitCount,missCount+hitCount))