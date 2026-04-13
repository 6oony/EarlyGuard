import os
import random
import pandas as pd

random.seed(42)

COMMON_SYSCALLS = [
    "open", "openat", "read", "write", "close",
    "fstat", "stat", "lstat", "newfstatat",
    "mmap", "munmap", "mprotect", "brk",
    "access", "getpid", "gettid", "clock_gettime",
    "gettimeofday", "futex", "lseek", "getrandom",
    "arch_prctl", "set_tid_address", "set_robust_list",
]

RANSOMWARE_BIASED = [
    "rename", "unlink", "ftruncate", "fsync", "fdatasync",
    "mkdir", "rmdir", "getdents", "readdir",
    "socket", "connect", "send", "recv",
]

BENIGN_BIASED = [
    "execve", "poll", "epoll_wait", "dup", "dup2", "pipe",
    "prlimit64", "wait4", "nanosleep", "ioctl",
]

NOISE_SYSCALLS = [
    "read", "write", "close", "mmap", "munmap", "fstat",
    "openat", "lseek", "clock_gettime", "futex",
    "stat", "access", "getpid", "brk",
]

CONFUSING_SYSCALLS = [
    "rename", "unlink", "execve", "poll", "socket", "connect",
    "dup", "dup2", "mkdir", "ioctl"
]


def weighted_sample(pool, k, low=0.7, high=2.0):
    weights = [random.uniform(low, high) for _ in pool]
    return random.choices(pool, weights=weights, k=k)


def maybe_inject_confusion(sequence, intensity=0.08):
    n = max(1, int(len(sequence) * intensity))
    for _ in range(n):
        idx = random.randint(0, len(sequence) - 1)
        sequence[idx] = random.choice(CONFUSING_SYSCALLS)
    return sequence


def build_startup_phase(length):
    return weighted_sample(COMMON_SYSCALLS, length, 0.9, 1.8)


def build_core_phase(label, length):
    if label == "ransomware":
        pool = COMMON_SYSCALLS * 5 + RANSOMWARE_BIASED * 3 + BENIGN_BIASED
    else:
        pool = COMMON_SYSCALLS * 5 + BENIGN_BIASED * 3 + RANSOMWARE_BIASED
    return weighted_sample(pool, length, 0.8, 1.9)


def build_noise_phase(length):
    return weighted_sample(NOISE_SYSCALLS + COMMON_SYSCALLS, length, 0.8, 1.6)


def inject_ransomware_burst(sequence):
    burst = random.choices(
        ["open", "read", "write", "rename", "unlink", "fsync", "fdatasync"],
        k=random.randint(3, 6)
    )
    insert_at = random.randint(max(5, len(sequence) // 5), min(len(sequence) - 3, len(sequence) // 2))
    sequence[insert_at:insert_at] = burst
    return sequence


def inject_benign_file_activity(sequence):
    burst = random.choices(
        ["open", "read", "write", "close", "lseek", "fstat", "mmap"],
        k=random.randint(3, 6)
    )
    insert_at = random.randint(max(5, len(sequence) // 5), min(len(sequence) - 3, len(sequence) // 2))
    sequence[insert_at:insert_at] = burst
    return sequence


def generate_sequence(label, length_range=(100, 220)):
    total_len = random.randint(*length_range)

    startup_len = max(20, int(total_len * random.uniform(0.22, 0.30)))
    core_len = max(45, int(total_len * random.uniform(0.42, 0.52)))
    noise_len = total_len - startup_len - core_len

    startup = build_startup_phase(startup_len)
    core = build_core_phase(label, core_len)
    noise = build_noise_phase(noise_len)

    sequence = startup + core + noise

    if label == "ransomware":
        sequence = inject_ransomware_burst(sequence)
    else:
        sequence = inject_benign_file_activity(sequence)

    confusion_level = random.uniform(0.05, 0.12)
    sequence = maybe_inject_confusion(sequence, intensity=confusion_level)

    if label == "ransomware" and random.random() < 0.30:
        for _ in range(random.randint(2, 5)):
            idx = random.randint(0, min(len(sequence) - 1, 40))
            sequence[idx] = random.choice(COMMON_SYSCALLS + BENIGN_BIASED)

    if label == "benign" and random.random() < 0.25:
        for _ in range(random.randint(2, 4)):
            idx = random.randint(0, min(len(sequence) - 1, 50))
            sequence[idx] = random.choice(RANSOMWARE_BIASED)

    return sequence


def generate_csv(sequence, filepath):
    df = pd.DataFrame(sequence)
    df.to_csv(filepath, index=False, header=False)


def main():
    dirs = {
        "dataset/ransomware_calls": "ransomware",
        "dataset/benign_calls": "benign",
    }

    for folder in dirs:
        os.makedirs(folder, exist_ok=True)
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                os.remove(os.path.join(folder, filename))

    for folder, label in dirs.items():
        for i in range(80):
            sequence = generate_sequence(label)
            path = os.path.join(folder, f"{label}_{i+1:03d}.csv")
            generate_csv(sequence, path)
        print(f"Generated 80 CSV files in '{folder}'")

    print("\\nDone. Re-run:")
    print("python train_model.py")


if __name__ == "__main__":
    main()
