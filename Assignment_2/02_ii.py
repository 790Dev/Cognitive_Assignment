arr = (45, 89.5, 76, 45.4, 89, 92, 58, 45)

lowest = 100

index = 0
i = 0
while(i<len(arr)):
    if(lowest>arr[i]):
        lowest = arr[i]
        index = i
    i+=1

print(f"The lowest value is {lowest} and index is {index}")