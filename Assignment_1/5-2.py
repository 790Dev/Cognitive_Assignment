n = int(input("enter the number "))

sum  = 0

for i in range(1,n+1,1):
    if(i%7==0 and i%9==0):
        sum+=i

print(f"Total sum of num divide by 7 and 9 is {sum}")
