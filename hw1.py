OUID = 113596441

def to_base_26(num):
    base_26 = ""
    while num > 0:
        remainder = num % 26
        num = num // 26
        base_26 = chr(65 + remainder) + base_26
    return base_26

def from_base_26(base_26):
    num = 0
    for i in range(len(base_26)):
        num += (ord(base_26[i]) - 65) * 26 ** (len(base_26) - i - 1)
    return num

def int_equivalent_of_char(char): # A = 0, B = 1, ..., Z = 25
    return ord(char) - 65

def char_equivalent_of_int(num): # 0 = A, 1 = B, ..., 25 = Z
    return chr(num + 65)

def mult_base_26(num1, num2): # this starts at the highest power of 26
    sum = 0
    for i in range(len(num1)):
        a = int_equivalent_of_char(num1[i])
        for j in range(len(num2)):
            b = int_equivalent_of_char(num2[j])
            sum += (a * 26 ** (len(num1) - i - 1)) * (b * 26 ** (len(num2) - j - 1))
    return to_base_26(sum)

def div_base_26(num1, num2):
    a = from_base_26(num1)
    b = from_base_26(num2)
    return to_base_26(a // b), to_base_26(a % b)

# a
OUID_26 = to_base_26(OUID)
print(OUID_26)

## testing a
print(from_base_26(OUID_26))
print()

# b
res_mult = mult_base_26(OUID_26, "DALLAS")
print(res_mult)

## testing b
print(from_base_26(res_mult))
print(OUID * from_base_26("DALLAS"))
print()

# c
res_div, res_mod = div_base_26(res_mult, "OKC")
print(res_div, "r", res_mod)

## testing c
print(from_base_26(res_div), "r", from_base_26(res_mod))
print(OUID * from_base_26("DALLAS") // from_base_26("OKC"), "r", OUID * from_base_26("DALLAS") % from_base_26("OKC"))

