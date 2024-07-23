import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time
import argparse

# Constants
MAX_PASSWORD_LENGTH = 8
BATCH_SIZE = 10000000  # Increased batch size
PRIME1 = 2**31 - 1  # Mersenne prime
PRIME2 = 2**61 - 1  # Larger Mersenne prime

def improved_hash(password):
    """Improved GPU-friendly hash function with better collision resistance."""
    hash_value1 = torch.zeros(password.shape[0], dtype=torch.int32, device=password.device)
    hash_value2 = torch.zeros(password.shape[0], dtype=torch.int64, device=password.device)
    for i in range(password.shape[1]):
        hash_value1 = (hash_value1 * 31 + password[:, i]) % PRIME1
        hash_value2 = (hash_value2 * 37 + password[:, i]) % PRIME2
    return hash_value1, hash_value2

def generate_password_batch(start, batch_size, length, device):
    """Generate a batch of passwords of a given length using ASCII values."""
    passwords = torch.zeros((batch_size, length), dtype=torch.int32, device=device)
    indices = torch.arange(start, start + batch_size, dtype=torch.int64, device=device)
    
    charset = torch.tensor(
        list(range(ord('a'), ord('z')+1)) + 
        list(range(ord('A'), ord('Z')+1)) + 
        list(range(ord('0'), ord('9')+1)), 
        dtype=torch.int32, device=device
    )
    charset_size = len(charset)
    
    for i in range(length - 1, -1, -1):
        passwords[:, i] = charset[indices % charset_size]
        indices //= charset_size
    
    return passwords

def attempt_to_string(attempt, length):
    """Convert a tensor of ASCII values to a string."""
    return ''.join(chr(int(i.item())) for i in attempt[:length])

def verify_hash(password_str):
    """Verify hash calculation for a single password string."""
    hash_value1, hash_value2 = 0, 0
    for char in password_str:
        hash_value1 = (hash_value1 * 31 + ord(char)) % PRIME1
        hash_value2 = (hash_value2 * 37 + ord(char)) % PRIME2
    return hash_value1, hash_value2

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def crack_password_distributed(rank, world_size, target_password, batch_size=BATCH_SIZE):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Using {world_size} GPUs")
        print(f"Target password: {target_password}")

    # Convert target password to tensor and compute its hash
    target_password_tensor = torch.tensor([[ord(c) for c in target_password]], dtype=torch.int32, device=device)
    target_hash1, target_hash2 = improved_hash(target_password_tensor)
    target_hash1, target_hash2 = target_hash1[0].item(), target_hash2[0].item()

    if rank == 0:
        print(f"Target hash: ({target_hash1}, {target_hash2})")
        print(f"Verifying target hash calculation:")
        print(f"Target password tensor: {target_password_tensor}")
        recalc_hash1, recalc_hash2 = improved_hash(target_password_tensor)
        print(f"Recalculated hash: ({recalc_hash1[0].item()}, {recalc_hash2[0].item()})")
        py_hash1, py_hash2 = verify_hash(target_password)
        print(f"Python verification: ({py_hash1}, {py_hash2})")

    start_time = time.time()

    charset_size = 26 + 26 + 10  # lowercase + uppercase + digits
    total_checked = 0

    for length in range(1, len(target_password) + 1):
        if rank == 0:
            print(f"Checking passwords of length {length}")
        total_combinations = charset_size ** length
        
        for start in range(rank, total_combinations, world_size * batch_size):
            end = min(start + batch_size, total_combinations)
            current_batch_size = end - start
            
            passwords_batch = generate_password_batch(start, current_batch_size, length, device)
            hashes1, hashes2 = improved_hash(passwords_batch)
            
            found = ((hashes1 == target_hash1) & (hashes2 == target_hash2)).nonzero(as_tuple=True)[0]
            if found.numel() > 0:
                for idx in found:
                    index = idx.item()
                    found_password = attempt_to_string(passwords_batch[index], length)
                    if found_password == target_password:
                        end_time = time.time()
                        if rank == 0:
                            print(f"Password found: {found_password}")
                            print(f"Time taken: {end_time - start_time:.4f} seconds")
                            print(f"Total passwords checked: {total_checked}")
                        cleanup()
                        return
                if rank == 0:
                    print(f"Hash collision detected. Continuing search...")
                    colliding_password = attempt_to_string(passwords_batch[found[0]], length)
                    print(f"Colliding password: {colliding_password}")
                    print(f"Verifying collision:")
                    ch1, ch2 = verify_hash(target_password)
                    print(f"Hash of '{target_password}': ({ch1}, {ch2})")
                    ch1, ch2 = verify_hash(colliding_password)
                    print(f"Hash of '{colliding_password}': ({ch1}, {ch2})")
            
            total_checked += current_batch_size * world_size
            if rank == 0 and total_checked % (100 * batch_size * world_size) == 0:
                print(f"Checked {total_checked} passwords...")

        # Synchronize GPUs after each password length
        dist.barrier()

    end_time = time.time()
    if rank == 0:
        print("Password not found")
        print(f"Time taken: {end_time - start_time:.4f} seconds")
        print(f"Total passwords checked: {total_checked}")

    cleanup()

def run_distributed(target_password):
    world_size = torch.cuda.device_count()
    mp.spawn(crack_password_distributed,
             args=(world_size, target_password),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Password Cracker")
    parser.add_argument("password", help="The password to crack")
    args = parser.parse_args()

    run_distributed(args.password)