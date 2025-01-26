def Addodd(n):
    sum = 0
    for i in range(1,n+1,1):
        if(i%2!=0):
            sum+=i
    
    return sum

n = int(input("Enter the number to get the sum of odd num "))
sum = Addodd(n)
print(f"The sum of odd num for 1 to {n} is {sum}")



