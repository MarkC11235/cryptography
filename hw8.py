AES_MIX_COLUMNS_MATRIX = [
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2]
]

# The steps for this multiplication were found at 
# https://en.wikipedia.org/wiki/Finite_field_arithmetic#Rijndael%27s_finite_field
def gf_multiply(a, b):
    # loop 8 times (once for each bit)
    p = 0  # product
    for _ in range(8):
        # if rightmost bit of b is 1, XOR the product with a
        if b & 1:
            p ^= a
        # shift b to the right by 1
        b >>= 1
        # check if the leftmost bit of a is 1 (this is the carry)
        carry = a & 0x80
        # shift a to the left by 1
        a <<= 1
        # if there was a carry, XOR a with 0x1B (the irreducible polynomial)
        if carry:
            a ^= 0x1B
    return p

def mix_columns(state):
    new_state = [[0] * 4 for _ in range(4)]  # 4x4 matrix filled with zeros
    # normal matrix multiplication
    for col in range(4):  
        for row in range(4): 
            new_state[row][col] = (
                # Here we use xor to combine the results of the multiplication
                gf_multiply(AES_MIX_COLUMNS_MATRIX[row][0], state[0][col]) ^
                gf_multiply(AES_MIX_COLUMNS_MATRIX[row][1], state[1][col]) ^
                gf_multiply(AES_MIX_COLUMNS_MATRIX[row][2], state[2][col]) ^
                gf_multiply(AES_MIX_COLUMNS_MATRIX[row][3], state[3][col])
            )
    return new_state

state_matrix = [
    [ord('O'), ord('K'), ord('L'), ord('A')],
    [ord('H'), ord('O'), ord('M'), ord('A')],
    [ord('I'), ord('L'), ord('L'), ord('I')],
    [ord('N'), ord('O'), ord('I'), ord('S')]
]

new_state = mix_columns(state_matrix)

# int -> hex
hex_state = [[hex(val) for val in row] for row in new_state]

print("New state after MixColumns transformation:")
for row in hex_state:
    print(row)

print("\nASCII values of the new state matrix:")
for row in new_state:
    print([chr(val) for val in row])