# File: improved_hash_diagnostic.py
PRIME = 2**61 - 1  # Mersenne prime

def improved_python_hash(password):
    hash_value = 0
    for char in password:
        hash_value = (hash_value * 31 + ord(char)) % PRIME
    return hash_value

def test_improved_hash_function():
    test_cases = ['Pass1', 'OEWX1', 'a', 'b', 'aa', 'ab', 'ba', 'abc', 'acb', 'password', 'Password1']
    
    print("Testing improved hash function:")
    for password in test_cases:
        hash_value = improved_python_hash(password)
        print(f"Hash of '{password}': {hash_value}")

    # Test for uniqueness
    all_hashes = [improved_python_hash(password) for password in test_cases]
    unique_hashes = set(all_hashes)
    print(f"\nNumber of unique hashes: {len(unique_hashes)} out of {len(all_hashes)} total")

if __name__ == "__main__":
    test_improved_hash_function()