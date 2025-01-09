import argparse
import json
import os
import hashlib
import random

import numpy as np
from tqdm.auto import tqdm, trange


# Класс для работы с матрицами по модулю
class ModuloMatrix:
    """Operations on matrices modulo a given modulus."""

    def __init__(self, modulus: int):
        """Инициализация с заданным модулем."""
        self.mod = modulus

    def mod_inv(self, a: int) -> int:
        """Вычисление мультипликативной обратной величины a mod self.mod."""
        a = a % self.mod
        if a == 0:
            raise ValueError("No inverse for 0.")  # Для 0 нет обратного элемента
        # Расширенный алгоритм Евклида
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
        """Умножение двух квадратных матриц A и B по модулю self.mod."""
        A = np.array(A, dtype=int)
        B = np.array(B, dtype=int)

        result = (A @ B) % self.mod  # Умножение с остатком
        return tuple(tuple(int(v) for v in row) for row in result)

    def mat_inv(self, A, postfix: str = ""):
        """Вычисление обратной матрицы по модулю."""
        A = np.array(A, dtype=int)
        n = A.shape[0]
        I = np.eye(n, dtype=int)  # Единичная матрица
        mod = self.mod

        for i in trange(n, desc=f"Inverting {postfix}"):
            # Нахождение опорного элемента
            pivot = -1
            for r in range(i, n):
                if A[r, i] % mod != 0:
                    pivot = r
                    break
            if pivot == -1:
                raise ValueError("Matrix not invertible")  # Матрица необратима

            # Замена строк
            if pivot != i:
                A[[i, pivot]] = A[[pivot, i]]
                I[[i, pivot]] = I[[pivot, i]]

            # Нормализация строки с опорным элементом
            inv_pivot = self.mod_inv(A[i, i])
            A[i, :] = (A[i, :] * inv_pivot) % mod
            I[i, :] = (I[i, :] * inv_pivot) % mod

            # Устранение влияния текущей строки на остальные
            factors = A[:, i].copy()
            factors[i] = 0
            nonzero_rows = (factors != 0)

            A[nonzero_rows] = (A[nonzero_rows] - factors[nonzero_rows, None] * A[i]) % mod
            I[nonzero_rows] = (I[nonzero_rows] - factors[nonzero_rows, None] * I[i]) % mod

        return tuple(tuple(row) for row in I)


# Класс для шифрования с использованием некомутационных матриц
class MatrixCrypto:
    """Noncommutative matrix-based encryption using a given matrix dimension."""

    def __init__(self, dimension: int, modulus: int = 37):
        """Инициализация с размерностью матрицы и модулем."""
        self.dim = dimension  # Размерность матрицы
        self.modulus = modulus  # Модуль
        self.mm = ModuloMatrix(modulus)  # Экземпляр ModuloMatrix
        self.rng = random.SystemRandom()  # Случайный генератор для безопасности

    def conjugate(self, x, a, apostfix: str = "", precomputed_inv=None):
        """Вычисление сопряжения x^a = a^{-1} x a."""
        a_inv = self.mm.mat_inv(a, apostfix) if precomputed_inv is None else precomputed_inv
        left = self.mm.mat_mul(a_inv, x)
        return self.mm.mat_mul(left, a), a_inv

    def random_jordan_cell(self):
        """Создание случайной жордановой клетки."""
        M = [[0] * self.dim for _ in range(self.dim)]
        lmda = self.rng.randrange(1, self.modulus)  # Случайное значение на диагонали
        for i in range(self.dim):
            M[i][i] = lmda
            if i > 0:
                M[i - 1][i] = 1  # Верхняя единичная поддиагональ
        return tuple(tuple(row) for row in M)

    def random_upper_triangle(self):
        """Создание случайной верхнетреугольной матрицы с одинаковыми диагональными элементами."""
        M = [[0] * self.dim for _ in range(self.dim)]
        diag_el = [self.rng.randrange(1, self.modulus) for _ in range(self.dim)]
        for i in range(self.dim):
            M[i][i] = diag_el[0]
            for j in range(i + 1, self.dim):
                M[i][j] = diag_el[j - i]
        return tuple(tuple(row) for row in M)

    def hash_matrix(self, M):
        """Хэширование матрицы и получение дайджеста."""
        arr = np.array(M)
        data = arr.tobytes()
        return hashlib.sha256(data).digest()

    def xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """Побитовый XOR двух последовательностей байтов."""
        return bytes(x ^ y for x, y in zip(a, b))

    def generate_random_matrix(self):
        """Генерация случайной матрицы."""
        M = []
        for __ in trange(self.dim, desc="Generate x"):
            row = [self.rng.randrange(0, self.modulus) for _ in range(self.dim)]
            M.append(row)
        return M

    def generate_keys(self, keydir: str, x_file: str, sk_file: str, pk_file: str):
        """Генерация ключей и сохранение их в файлы."""
        if os.path.exists(keydir) and os.listdir(keydir):
            ans = input(f"Directory '{keydir}' already exists. Overwrite? (y/n): ").strip().lower()
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

        x = self.generate_random_matrix()
        with open(os.path.join(keydir, x_file), "w") as f:
            json.dump(x, f)

        b = self.random_upper_triangle()
        z, _ = self.conjugate(tuple(tuple(row) for row in x), b, "secret key b")

        with open(os.path.join(keydir, sk_file), "w") as f:
            json.dump(b, f)
        with open(os.path.join(keydir, pk_file), "w") as f:
            json.dump(z, f)

    def encrypt_message(self, keydir: str, pk_file: str, x_file: str, message: str, output_file: str):
        """Шифрование сообщения с использованием открытого ключа."""
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
        """Расшифрование сообщения с использованием закрытого ключа."""
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
    """Главная функция для управления командами."""
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
