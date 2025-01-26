def SumPrime(n):
    sum = 0 
    for i in range(2,n+1,1):
        flag = 1 ## 1 is for prime
        for j in range(2,(i)//2+1,1):
            if(i%j==0):
                flag=0
                break
        if(flag==1):
            sum+=i

    
    return sum

n = int(input("Enter the number to get sum of prime num"))

sum  = SumPrime(n)
print(f" sum id {sum}")