import numpy
from function import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
from numpy import loadtxt
import random

f=open("AllTextV1.txt","r")
readLine=f.readlines() #ใช้เเค่ตอน read

f1=open("fold1.txt","w") #เอาไว้เขียนลงไฟล์
f2=open("fold2.txt","w") #เอาไว้เขียนลงไฟล์
f3=open("fold3.txt","w") #เอาไว้เขียนลงไฟล์
f4=open("fold4.txt","w") #เอาไว้เขียนลงไฟล์
f5=open("fold5.txt","w") #เอาไว้เขียนลงไฟล์
f6=open("fold6.txt","w") #เอาไว้เขียนลงไฟล์
f7=open("fold7.txt","w") #เอาไว้เขียนลงไฟล์
f8=open("fold8.txt","w") #เอาไว้เขียนลงไฟล์
f9=open("fold9.txt","w") #เอาไว้เขียนลงไฟล์
f10=open("fold10.txt","w") #เอาไว้เขียนลงไฟล์
f1.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f2.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f3.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f4.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f5.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f6.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f7.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f8.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f9.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
f10.write("Age	SystolicBP	DiastolicBP	BS	BodyTemp	HeartRate	RiskLevel\n")
#if ((X[i].count('<=30')==1)): #ตอนเช็ค

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f1.write(readLine[rand])
    #print("%d"%(rand))
for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f2.write(readLine[rand])
    
for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f3.write(readLine[rand])
    
for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f4.write(readLine[rand])

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f5.write(readLine[rand])

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f6.write(readLine[rand])

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f7.write(readLine[rand])

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f8.write(readLine[rand])

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f9.write(readLine[rand])

for i in range(0,100):
    rand  = (random.randint(1, 1014))
    f10.write(readLine[rand])