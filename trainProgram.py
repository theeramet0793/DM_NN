import numpy
from function import * # ถ้า import * คือเอาทุก function ในไฟล์ NNfunction
from numpy import loadtxt

f=open("foldTrain.txt","r")
readLine=f.readlines() #ใช้เเค่ตอน read

#f1=open("buycomL2left.txt","w") เอาไว้เขียนลงไฟล์
#if ((X[i].count('<=30')==1)): ตอนเช็ค

w7 = ([0.9,0.5,0.5,0.4,0.2,0.1,-0.4])
w8 = ([0.5,0.5,0.3,0.6,0.1,0.3,-0.4])
w9 = ([0.1,0.2,0.3,0.4,0.1,0.1,-0.4])
w10 = ([0.5,0.2,0.5,-0.9])
w11 = ([0.2,0.8,0.1,-0.5])
w12 = ([0.5,0.6,0.3,-0.1])



l = -0.9 # learning rate
epoch = 0
err10 = 1000
err11 = 1000
err12 = 1000

while(True):
    for i in range(1,901) :
        separate = readLine[i].split(sep="\t", maxsplit=7)
        #print("%s %s %s %s %s %s %s"%(separate[0],separate[1],separate[2],separate[3],separate[4],separate[5],separate[6]))
        d10 = 11 #desire output
        d11 = 11 #desire output
        d12 = 11 #desire output
        
        X  = ([int(separate[0]),int(separate[1]),int(separate[2]),int(separate[3]),int(separate[4]),int(separate[5]),int(separate[6])])
        
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
        #print("%d Class = %d%d%d"%(X[6],d10,d11,d12))
        
        
        #forward pass
        print("\n========================= Round %d ========================="%i)
        print("\n>>>>>>>>>>>>>>>>>>>> Forward pass")
        o7=Nout(X,w7)
        y7=sigmoid(o7)
        print("\nOutput from node 7 is: %8.3f, Y from node 7 is: %8.3f" % (o7,y7))
        o8=Nout(X,w8)
        y8=sigmoid(o8)
        print("\nOutput from node 8 is: %8.3f, Y from node 8 is: %8.3f" % (o8,y8))
        o9=Nout(X,w9)
        y9=sigmoid(o9)
        print("\nOutput from node 9 is: %8.3f, Y from node 9 is: %8.3f" % (o9,y9))
        x10=([y7,y8,y9,d10])
        x11=([y7,y8,y9,d11])
        x12=([y7,y8,y9,d12])
        
        o10=Nout(x10,w10)
        o11=Nout(x11,w11)
        o12=Nout(x12,w12)
        y10=sigmoid(o10)
        y11=sigmoid(o11)
        y12=sigmoid(o12)
        print("\nOutput from node 10 is: %8.3f, Y from node 10 is: %8.3f" % (o10,y10))
        print("\nOutput from node 11 is: %8.3f, Y from node 11 is: %8.3f" % (o11,y11))
        print("\nOutput from node 12 is: %8.3f, Y from node 12 is: %8.3f" % (o12,y12))
        
        
        #backpropagation
        #node 12
        print("\n>>>>>>>>>>>>>>>>>>>> Back propagation & calculate new Weights and Biases")
        e12=d12-y12
        g12=gradOut(e12,y12)
        dw712=deltaw(l,g12,y7)
        w712n=w12[0]+dw712
        dw812=deltaw(l,g12,y8)
        w812n=w12[1]+dw812
        dw912=deltaw(l,g12,y9)
        w912n=w12[2]+dw912
        db12=deltaw(l,g12,1)
        b12n=w12[2]+db12
        print("\nNew w712 is %8.3f, New w812 is:%8.3f, New w912 is:%8.3f, New bias 12 is %8.3f"%(w712n,w812n,w912n,b12n))
        
        #node 11
        print("\n>>>>>>>>>>>>>>>>>>>> Back propagation & calculate new Weights and Biases")
        e11=d11-y11
        g11=gradOut(e11,y11)
        dw711=deltaw(l,g11,y7)
        w711n=w11[0]+dw711
        dw811=deltaw(l,g11,y8)
        w811n=w11[1]+dw811
        dw911=deltaw(l,g11,y9)
        w911n=w11[2]+dw911
        db11=deltaw(l,g11,1)
        b11n=w11[2]+db11
        print("\nNew w711 is %8.3f, New w811 is:%8.3f, New w911 is:%8.3f, New bias 11 is %8.3f"%(w711n,w811n,w911n,b11n))
        
        #node 10
        print("\n>>>>>>>>>>>>>>>>>>>> Back propagation & calculate new Weights and Biases")
        e10=d10-y10
        g10=gradOut(e10,y10)
        dw710=deltaw(l,g10,y7)
        w710n=w10[0]+dw710
        dw810=deltaw(l,g10,y8)
        w810n=w10[1]+dw810
        dw910=deltaw(l,g10,y9)
        w910n=w10[2]+dw910
        db10=deltaw(l,g10,1)
        b10n=w10[2]+db10
        print("\nNew w710 is %8.3f, New w810 is:%8.3f, New w910 is:%8.3f, New bias 10 is %8.3f"%(w710n,w810n,w910n,b10n))
        
        #node9
        g9=gradH(y9,g10*(w10[2])+g11*(w11[2])+g12*(w12[2]))
        dw19=deltaw(l,g9,X[0])
        w19n=w9[0]+dw19
        dw29=deltaw(l,g9,X[1])
        w29n=w9[1]+dw29
        dw39=deltaw(l,g9,X[2])
        w39n=w9[2]+dw39
        dw49=deltaw(l,g9,X[3])
        w49n=w9[3]+dw49
        dw59=deltaw(l,g9,X[4])
        w59n=w9[4]+dw59
        dw69=deltaw(l,g9,X[5])
        w69n=w9[5]+dw69

        db9=deltaw(l,g9,1)
        b9n=w9[3]+db9
        print("\nNew w19 is %8.3f, New w29 is:%8.3f, New w39 is:%8.3f, New w49 is:%8.3f, New w59 is:%8.3f, New w69 is:%8.3f, New bias 9 is %8.3f"%(w19n,w29n,w39n,w49n,w59n,w69n,b9n))
        

        #node8
        g8=gradH(y8,g10*(w10[1])+g11*(w11[1])+g12*(w12[1]))
        dw18=deltaw(l,g8,X[0])
        w18n=w8[0]+dw18
        dw28=deltaw(l,g8,X[1])
        w28n=w8[1]+dw28
        dw38=deltaw(l,g8,X[2])
        w38n=w8[2]+dw38
        dw48=deltaw(l,g8,X[3])
        w48n=w8[3]+dw48
        dw58=deltaw(l,g8,X[4])
        w58n=w8[4]+dw58
        dw68=deltaw(l,g8,X[5])
        w68n=w8[5]+dw68

        db8=deltaw(l,g8,1)
        b8n=w8[3]+db8
        print("\nNew w18 is %8.3f, New w28 is:%8.3f, New w38 is:%8.3f, New w48 is:%8.3f, New w58 is:%8.3f, New w68 is:%8.3f, New bias 8 is %8.3f"%(w18n,w28n,w38n,w48n,w58n,w68n,b8n))
        
        #node7
        g7=gradH(y7,g10*(w10[0])+g11*(w11[0])+g12*(w12[0]))
        dw17=deltaw(l,g7,X[0])
        w17n=w7[0]+dw17
        dw27=deltaw(l,g7,X[1])
        w27n=w7[1]+dw27
        dw37=deltaw(l,g7,X[2])
        w37n=w7[2]+dw37
        dw47=deltaw(l,g7,X[3])
        w47n=w7[3]+dw47
        dw57=deltaw(l,g7,X[4])
        w57n=w7[4]+dw57
        dw67=deltaw(l,g7,X[5])
        w67n=w7[5]+dw67

        db7=deltaw(l,g7,1)
        b7n=w7[3]+db7
        print("\nNew w17 is %8.3f, New w27 is:%8.3f, New w37 is:%8.3f, New w47 is:%8.3f, New w57 is:%8.3f, New w67 is:%8.3f, New bias 7 is %8.3f"%(w17n,w27n,w37n,w47n,w57n,w67n,b7n))
        
        w7=([w17n,w27n,w37n,w47n,w57n,w67n,b7n])
        w8=([w18n,w28n,w38n,w48n,w58n,w68n,b8n])
        w9=([w19n,w29n,w39n,w49n,w59n,w69n,b9n])
        w10=([w710n,w810n,w910n,b10n])
        w11=([w711n,w811n,w911n,b11n])
        w12=([w712n,w812n,w912n,b12n])
        
        
        print("\n>>>>>>>>>>>>>>>>>>>> error10 = %8.3f , error11 = %8.3f , error12 = %8.3f "%(e10,e11,e12))
        err10 = e10
        err11 = e11
        err12 = e12
        
    epoch = epoch + 1
    print("========================================= EPOCH = %d ==========================================\n"%epoch)
    if((abs(err10)<0.100)and(abs(err11)<0.100)and(abs(err12)<0.100)or(epoch==5)):
        f1=open("forKeepWandB.txt","w") #เอาไว้เขียนลงไฟล์
        f1.write("w7 = ([%8.3f,     %8.3f,     %8.3f,    %8.3f,    %8.3f,    %8.3f,    %8.3f])\n"%(w17n,w27n,w37n,w47n,w57n,w67n,b7n))
        f1.write("w8 = ([%8.3f,     %8.3f,     %8.3f,    %8.3f,    %8.3f,    %8.3f,    %8.3f])\n"%(w18n,w28n,w38n,w48n,w58n,w68n,b8n))
        f1.write("w9 = ([%8.3f,     %8.3f,     %8.3f,    %8.3f,    %8.3f,    %8.3f,    %8.3f])\n"%(w19n,w29n,w39n,w49n,w59n,w69n,b9n))
        f1.write("w10=([%8.3f,  %8.3f,  %8.3f,  %8.3f])\n"%(w710n,w810n,w910n,b10n))
        f1.write("w11=([%8.3f,  %8.3f,  %8.3f,  %8.3f])\n"%(w711n,w811n,w911n,b11n))
        f1.write("w12=([%8.3f,  %8.3f,  %8.3f,  %8.3f])\n"%(w712n,w812n,w912n,b12n))
        break

    
"""
W4=([0.2,0.4,-0.5,-0.4])
W5=([-0.3,0.1,0.2,0.2])
d6=1 #desire output 
l=-0.9 # learning rate
round = 1
check = True
count = 1 
W6=([-0.3,-0.2,0.1])


while(check):
    #forward pass
    print("\n=========================Round %d ========================="%count)
    print("\n******Forward pass****** ")
    o4=Nout(X,W4)
    y4=sigmoid(o4)
    print("\nOutput from node 4 is: %8.3f, Y from node 4 is: %8.3f" % (o4,y4))
    o5=Nout(X,W5)
    y5=sigmoid(o5)
    print("\nOutput from node 5 is: %8.3f, Y from node 5 is: %8.3f" % (o5,y5))
    X6=([y4,y5,1])
    
    o6=Nout(X6,W6)
    y6=sigmoid(o6)
    print("\nOutput from node 6 is: %8.3f, Y from node 5 is: %8.3f" % (o6,y6))



    #backpropagation
    #node 6
    print("\n*** Back propagation & calculate new Weights and Biases****")
    e6=d6-y6
    g6=gradOut(e6,y6)
    dw46=deltaw(l,g6,y4)
    w46n=W6[0]+dw46
    dw56=deltaw(l,g6,y5)
    w56n=W6[1]+dw56
    db6=deltaw(l,g6,1)
    b6n=W6[2]+db6
    print("\nNew w46 is %8.3f, New w56 is:%8.3f, New bias 6 is %8.3f"%(w46n,w56n,b6n))

    #node5
    e6=d6-y6
    g5=gradH(y5,g6*(W6[1]))
    dw15=deltaw(l,g5,X[0])
    w15n=W5[0]+dw15
    dw25=deltaw(l,g5,X[1])
    w25n=W5[1]+dw25
    dw35=deltaw(l,g5,X[2])
    w35n=W5[2]+dw35

    db5=deltaw(l,g5,1)
    b5n=W5[3]+db5
    print("\nNew w15 is %8.3f, New w25 is:%8.3f, New w35 is:%8.3f, New bias 5 is %8.3f"%(w15n,w25n,w35n,b5n))

    #node4
    e6=d6-y6
    g4=gradH(y4,g6*(W6[0]))
    dw14=deltaw(l,g4,X[0])
    w14n=W4[0]+dw14
    dw24=deltaw(l,g4,X[1])
    w24n=W4[1]+dw24
    dw34=deltaw(l,g4,X[2])
    w34n=W4[2]+dw34

    db4=deltaw(l,g4,1)
    b4n=W4[3]+db4
    print("\nNew w14 is %8.3f, New w24 is:%8.3f, New w34 is:%8.3f, New bias 4 is %8.3f"%(w14n,w24n,w34n,b4n))
    print("\n ****** Error = %8.3f ******"%e6)
    
    count+=1
    if(e6<=0.10099999):
        check = False
    else:
        W4=([w14n,w24n,w34n,b4n])
        W5=([w15n,w25n,w35n,b5n])
        W6=([w46n,w56n,b6n])
"""