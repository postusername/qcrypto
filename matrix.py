import argparse
import json
import os
import hashlib
import random

import numpy as np
from tqdm.auto import tqdm, trange


class ModuloMatrix:
    """Operations on matrices modulo a given modulus."""

    def __init__(self, modulus: int):
        """Initialize with a given modulus."""
        self.mod = modulus

    def mod_inv(self, a: int) -> int:
        """Compute multiplicative inverse of a mod self.mod."""
        a = a % self.mod
        if a == 0:
            raise ValueError("No inverse for 0.")
        # Extended Euclidean algorithm
        t, newt = 0, 1
        r, newr = self.mod, a
        while newr != 0:
            q = r // newr
            t, newt = newt, t - q * newt
            r, newr = newr, r - q * newr
        if t < 0:
            t += self.mod
        return t

    def mat_mul(self, A, B):
        """Multiply two n x n matrices A and B modulo self.mod."""
        A = np.array(A, dtype=int)
        B = np.array(B, dtype=int)

        result = (A @ B) % self.mod
        return tuple(tuple(int(v) for v in row) for row in result)

    def mat_inv(self, A, postfix: str = ""):
        A = np.array(A, dtype=int)
        n = A.shape[0]
        I = np.eye(n, dtype=int)
        mod = self.mod

        for i in trange(n, desc=f"Inverting {postfix}"):
            # Find pivot
            pivot = -1
            for r in range(i, n):
                if A[r, i] % mod != 0:
                    pivot = r
                    break
            if pivot == -1:
                raise ValueError("Matrix not invertible")

            # Swap if needed
            if pivot != i:
                A[[i, pivot]] = A[[pivot, i]]
                I[[i, pivot]] = I[[pivot, i]]

            # Normalize pivot row
            inv_pivot = self.mod_inv(A[i, i])
            A[i, :] = (A[i, :] * inv_pivot) % mod
            I[i, :] = (I[i, :] * inv_pivot) % mod

            # Eliminate in other rows
            # Вместо цикла по всем строкам, используем векторизацию
            factors = A[:, i].copy()
            factors[i] = 0
            nonzero_rows = (factors != 0)

            # Для всех строк, где факторы не ноль, вычитаем factor * pivot_row
            A[nonzero_rows] = (A[nonzero_rows] - factors[nonzero_rows, None] * A[i]) % mod
            I[nonzero_rows] = (I[nonzero_rows] - factors[nonzero_rows, None] * I[i]) % mod

        return tuple(tuple(row) for row in I)


class MatrixCrypto:
    """Noncommutative matrix-based encryption using a given matrix dimension."""

    def __init__(self, dimension: int, modulus: int = 37):
        """Initialize with matrix dimension and modulus."""
        self.dim = dimension
        self.modulus = modulus
        self.mm = ModuloMatrix(modulus)
        self.rng = random.SystemRandom()

    def conjugate(self, x, a, apostfix: str = "", precomputed_inv=None):
        """Compute conjugation x^a = a^{-1} x a."""
        a_inv = self.mm.mat_inv(a, apostfix) if precomputed_inv is None else precomputed_inv
        left = self.mm.mat_mul(a_inv, x)
        return self.mm.mat_mul(left, a), a_inv

    def random_jordan_cell(self):
        # random Jordan cell
        M = [[0]*self.dim for _ in range(self.dim)]
        lmda = self.rng.randrange(1, self.modulus)
        for i in range(self.dim):
            M[i][i] = lmda
            if i > 0:
                M[i-1][i] = 1
        return tuple(tuple(row) for row in M)

    def random_upper_triangle(self):
        # random upper-triangle with eq. diag
        M = [[0]*self.dim for _ in range(self.dim)]
        diag_el = [self.rng.randrange(1, self.modulus) for _ in range(self.dim)]
        for i in range(self.dim):
            M[i][i] = diag_el[0]
            for j in range(i+1, self.dim):
                M[i][j] = diag_el[j-i]
        return tuple(tuple(row) for row in M)

    def hash_matrix(self, M):
        """Hash an n x n matrix M and return a digest."""
        arr = np.array(M)
        data = arr.tobytes()
        return hashlib.sha256(data).digest()

    def xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte sequences."""
        return bytes(x ^ y for x, y in zip(a, b))

    def generate_random_matrix(self):
        M = []
        for __ in trange(self.dim, desc="Generate x"):
            row = []
            for _ in range(self.dim):
                row.append(self.rng.randrange(0, self.modulus))
            M.append(row)
        return M

    def generate_keys(self, keydir: str, x_file: str, sk_file: str, pk_file: str):
        """Generate new keys. If directory not empty, ask user to overwrite."""
        if os.path.exists(keydir) and os.listdir(keydir):
            ans = input(f"Directory '{
                        keydir}' already exists and may contain keys. Overwrite? (y/n): ").strip().lower()
            if ans != 'y':
                print("Key generation aborted.")
                return
            for filename in os.listdir(keydir):
                file_path = os.path.join(keydir, filename)
                try:
                    os.remove(file_path)
                except Exception:
                    pass

        os.makedirs(keydir, exist_ok=True)

        x_path = os.path.join(keydir, x_file)
        sk_path = os.path.join(keydir, sk_file)
        pk_path = os.path.join(keydir, pk_file)

        x = self.generate_random_matrix()
        with open(x_path, "w") as f:
            json.dump(x, f)

        b = self.random_upper_triangle()
        # Convert to tuple of tuples if needed
        x_tuple = tuple(tuple(row) for row in x)
        z, _ = self.conjugate(x_tuple, b, "secret key b")

        with open(sk_path, "w") as f:
            json.dump(b, f)
        with open(pk_path, "w") as f:
            json.dump(z, f)

    def encrypt_message(self, keydir: str, pk_file: str, x_file: str, message: str, output_file: str):
        """Encrypt a message using public key z and public element x."""
        pk_path = os.path.join(keydir, pk_file)
        x_path = os.path.join(keydir, x_file)
        out_path = output_file

        with open(pk_path, "r") as f:
            z = json.load(f)
        z = tuple(tuple(row) for row in z)

        with open(x_path, "r") as f:
            x = json.load(f)
        x = tuple(tuple(row) for row in x)

        r = self.random_upper_triangle()
        xr, r_inv = self.conjugate(x, r, "random element r")
        t, _ = self.conjugate(z, r, precomputed_inv=r_inv)
        ht = self.hash_matrix(t)  # 32 байта

        m_bytes = message.encode()

        # Расширяем ключ ht до длины сообщения
        key_stream = ht
        while len(key_stream) < len(m_bytes):
            key_stream += ht

        # XOR по всему сообщению
        ctext = self.xor_bytes(key_stream[:len(m_bytes)], m_bytes)

        # Convert matrices to lists for JSON
        xr_list = [list(row) for row in xr]
        with open(out_path, "w") as f:
            json.dump({
                "xr": xr_list,
                "ctext": list(ctext)
            }, f)

    def decrypt_message(self, keydir: str, sk_file: str, x_file: str, input_file: str, output_file: str):
        """Decrypt a message using secret key b and public element x."""
        sk_path = os.path.join(keydir, sk_file)
        x_path = os.path.join(keydir, x_file)
        in_path = input_file
        out_path = output_file

        with open(sk_path, "r") as f:
            b = json.load(f)
        b = tuple(tuple(row) for row in b)

        with open(x_path, "r") as f:
            x = json.load(f)
        x = tuple(tuple(row) for row in x)

        with open(in_path, "r") as f:
            data = json.load(f)
        xr = data["xr"]
        xr = tuple(tuple(row) for row in xr)
        ctext = bytes(data["ctext"])

        t, _ = self.conjugate(xr, b, "secret key b")
        ht = self.hash_matrix(t)  # 32 байта

        # Расширяем ht до длины ciphertext
        key_stream = ht
        while len(key_stream) < len(ctext):
            key_stream += ht

        m_bytes = self.xor_bytes(ctext, key_stream[:len(ctext)])
        message = m_bytes.decode()

        if out_path:
            with open(out_path, "w") as f:
                f.write(message)
        else:
            print("Decrypted message:", message)

        return message


def add_common_args(parser):
    """Add arguments that are common to multiple subcommands."""
    parser.add_argument("--keydir", default="keys",
                        help="Directory to store/load key files (default: ./keys)")
    parser.add_argument("--x", default="x.json",
                        help="Public element x filename (default: x.json)")


def main():
    parser = argparse.ArgumentParser(
        description="Noncommutative matrix-based encryption.")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--dim", type=int, default=2,
                        help="Matrix dimension (default: 2)")

    keygen_parser = subparsers.add_parser("keygen", help="Generate keys")
    add_common_args(keygen_parser)
    keygen_parser.add_argument("--sk", default="secret_key.json",
                               help="Secret key filename (default: secret_key.json)")
    keygen_parser.add_argument("--pk", default="public_key.json",
                               help="Public key filename (default: public_key.json)")

    enc_parser = subparsers.add_parser("encrypt", help="Encrypt a message")
    add_common_args(enc_parser)
    enc_parser.add_argument("--pk", default="public_key.json",
                            help="Public key filename (default: public_key.json)")
    enc_parser.add_argument("--msg", required=True, help="Message to encrypt")
    enc_parser.add_argument("--out", default="ciphertext.json",
                            help="Output file for the ciphertext (default: ciphertext.json)")

    dec_parser = subparsers.add_parser("decrypt", help="Decrypt a message")
    add_common_args(dec_parser)
    dec_parser.add_argument("--sk", default="secret_key.json",
                            help="Secret key filename (default: secret_key.json)")
    dec_parser.add_argument("--infile", default="ciphertext.json",
                            help="Input file with ciphertext (default: ciphertext.json)")
    dec_parser.add_argument(
        "--out", default="", help="Output file for the decrypted message (optional, if empty prints to console)")

    args = parser.parse_args()

    mc = MatrixCrypto(dimension=args.dim, modulus=37)

    if args.command == "keygen":
        mc.generate_keys(args.keydir, args.x, args.sk, args.pk)
    elif args.command == "encrypt":
        mc.encrypt_message(args.keydir, args.pk, args.x, args.msg, args.out)
    elif args.command == "decrypt":
        mc.decrypt_message(args.keydir, args.sk, args.x, args.infile, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
