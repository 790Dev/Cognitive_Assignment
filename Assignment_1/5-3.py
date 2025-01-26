n = int(input("Enter the range of num to check "))
sum = 0 

for i in range(2,n+1,1):
    flag = 1; ## 1 is for prime and 0 is for composite
    for j in range(2,(i)//2+1,1):
        if(i%j==0):
            flag = 0
            break
    
    if(flag==1):
        sum+=i
 
print(f"Total sum is {sum}")