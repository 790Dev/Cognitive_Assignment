import random as r 


def OddNum(list):
    sum = 0
    for i in list:
        if(i%2!=0):
            sum+=i
            print(f"{i} ")
    
    print(f"Sum of oddNum is {sum}")
    return


def EvenNum(list):
    sum = 0
    for i in list:
        if(i%2==0):
            sum+=i
            print(f"{i} ")
    
    print(f"Sum of EvenNum is {sum}")
    return

def PrimeNum(list):
    sum=0
    for i in list:
        flag=1
        for j in range(2,i//2+1,1):
            if(i%j==0):
                flag=0
                break
        if(flag==1):
            sum+=i
            print(f"{i} ")
    print(f"Sum of prime num is {sum}")









list = r.sample(range(100,901),100)

OddNum(list)


