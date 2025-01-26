arr = (45, 89.5, 76, 45.4, 89, 92, 58, 45)

## 1st part
highest = -1

i = 0 
index = 0
while(i<len(arr)):
    if(highest<arr[i]):
        highest = arr[i]
        index = i
    i = i+1

print(f"The maximum is {highest} and the index is index is {index}")
