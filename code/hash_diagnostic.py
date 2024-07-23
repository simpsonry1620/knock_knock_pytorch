CHARSET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
PRIME = 2**61 - 1  # Mersenne prime

def python_hash(password):
    hash_value = 1
    for char in password:
        char_index = CHARSET.index(char)
        hash_value = (hash_value * 31) % PRIME
        hash_value = (hash_value + char_index) % PRIME
    return hash_value

def test_hash_function():
    test_cases = ['Pass1', 'OEWX1', 'a', 'b', 'aa', 'ab', 'ba', 'abc', 'acb', 'password', 'Password1']
    
    print("Testing hash function:")
    for password in test_cases:
        hash_value = python_hash(password)
        print(f"Hash of '{password}': {hash_value}")

    # Test for uniqueness
    all_hashes = [python_hash(password) for password in test_cases]
    unique_hashes = set(all_hashes)
    print(f"\nNumber of unique hashes: {len(unique_hashes)} out of {len(all_hashes)} total")

if __name__ == "__main__":
    test_hash_function()