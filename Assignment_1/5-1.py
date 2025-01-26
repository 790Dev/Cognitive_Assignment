a = int(input("enter the first number "))
b = int(input("enter the b number "))
c = int(input("enter the c number "))

if(a>b):
    if(a>c):
        print(f"{a} is greatest")
    else:
        print(f"{c} is greatest")
elif(c>b):
    print(f"{c} is greatest ")
else:
    print(f"{b} is gratest")
    

        