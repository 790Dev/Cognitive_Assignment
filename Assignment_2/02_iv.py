def checkNum(tuple,num):
    i = 0
    while(i<len(tuple)):
        if(num==tuple[i]):
            print(f"{num} is present at index {i}")
            break
        i+=1
    print(f"{num} is not present  ")
    return 


arr = (45, 89.5, 76, 45.4, 89, 92, 58, 45) 

num = int(input("Enter the number "))
checkNum(arr,num)