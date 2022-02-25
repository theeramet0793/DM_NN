import numpy
from function import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
from numpy import loadtxt

#เวลาใช้ เเก้เเค่ชื่อไฟล์ก็พอ
fr1=open("fold2.txt","r")
readLine1=fr1.readlines() 

fr2=open("fold3.txt","r")
readLine2=fr2.readlines()

fr3=open("fold4.txt","r")
readLine3=fr3.readlines()

fr4=open("fold5.txt","r")
readLine4=fr4.readlines()

fr5=open("fold6.txt","r")
readLine5=fr5.readlines()

fr6=open("fold7.txt","r")
readLine6=fr6.readlines()

fr7=open("fold8.txt","r")
readLine7=fr7.readlines()

fr8=open("fold9.txt","r")
readLine8=fr8.readlines()

fr9=open("fold10.txt","r")
readLine9=fr9.readlines()

fw=open("foldTrain.txt","w") #เอาไว้เขียนลงไฟล์
fw.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")

for i in range(1,101):
    fw.write(readLine1[i])
    
for i in range(1,101):
    fw.write(readLine2[i])
    
for i in range(1,101):
    fw.write(readLine3[i])

for i in range(1,101):
    fw.write(readLine4[i])

for i in range(1,101):
    fw.write(readLine5[i])

for i in range(1,101):
    fw.write(readLine6[i])

for i in range(1,101):
    fw.write(readLine7[i])

for i in range(1,101):
    fw.write(readLine8[i])

for i in range(1,101):
    fw.write(readLine9[i])