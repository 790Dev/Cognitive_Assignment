A = {34, 56, 78, 90}
B = {78, 45, 90, 23}

print(f"Unique by both teams {A|B}")

print(f"Common in both {A&B}")

print(f"Score Exclusive to A {A-B}")
print(f"Score Exclusive to A {B-A}")

is_subset = A.issubset(B)
is_superset = B.issuperset(A)

print(f"Is B is subset of A {is_subset}")
print(f"Is A is superset of A  {is_subset}")


X = int(input("enter the num to remove from set A"))

if X in A:
    A.remove(X)
    print(f"{X} is removed ")
else:
    print(f"{X} is not present in set A ")

