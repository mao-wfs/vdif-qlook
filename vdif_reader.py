__all__ = ["get_spectrum"]


# standard library
import re
from datetime import datetime, timedelta
from pathlib import Path
from struct import Struct
from typing import Callable, Pattern


# dependent packages
import numpy as np


# constants
LITTLE_ENDIAN: str = "<"
UINT: str = "I"
SHORT: str = "h"
N_ROWS_VDIF_HEAD: int = 8
N_ROWS_CORR_HEAD: int = 64
N_ROWS_CORR_DATA: int = 512
N_UNITS_PER_SCAN: int = 64
N_BYTES_PER_UNIT: int = 1312
N_BYTES_PER_SCAN: int = 1312 * 64
TIME_PER_SCAN: float = 5e-3  # seconds
TIME_FORMAT: str = "%Y%j%H%M%S"
VDIF_PATTERN: Pattern = re.compile(r"\w+_(\d+)_\d.vdif")


# main features
def get_spectrum(
    path: Path,
    integ: float = 1.0,
    delay: float = 0.0,
    chbin: int = 1,
) -> np.ndarray:
    spectra = get_spectra(path, integ, delay)
    spectrum = integrate_spectra(spectra, chbin)
    return spectrum


# sub features
def get_spectra(path: Path, integ: float = 1.0, delay: float = 0.0) -> np.ndarray:
    n_scans = get_n_scans_from_time(path, delay)
    n_integ = int(integ / TIME_PER_SCAN)
    n_units = N_UNITS_PER_SCAN * n_integ
    n_chans = N_ROWS_CORR_DATA // 2

    if n_scans - n_integ < 0:
        raise ValueError("Not enough number of scans to integrate")

    byte_start = N_BYTES_PER_SCAN * (n_scans - n_integ)
    spectra = np.empty([n_units, n_chans], dtype=complex)

    with open(path, "rb") as f:
        f.seek(byte_start)
        for i in range(n_units):
            read_vdif_head(f)
            read_corr_head(f)
            corr_data = read_corr_data(f)
            spectra[i] = parse_corr_data(corr_data)

    return spectra.reshape([n_integ, N_UNITS_PER_SCAN * n_chans])


def integrate_spectra(spectra: np.ndarray, chbin: int = 1) -> np.ndarray:
    spectrum = spectra.mean(0)
    return spectrum.reshape([len(spectrum) // chbin, chbin]).mean(1)


def get_n_scans_from_time(path: Path, delay: float = 0.0) -> int:
    match = VDIF_PATTERN.search(path.name)

    if match is None:
        raise ValueError("Cannot parse start time from file name.")

    t_start = datetime.strptime(match.groups()[0], TIME_FORMAT)
    t_now = datetime.utcnow() - timedelta(seconds=delay)

    return int((t_now - t_start).total_seconds() / TIME_PER_SCAN)


# struct readers
def make_binary_reader(n_rows: int, dtype: str) -> Callable:
    struct = Struct(LITTLE_ENDIAN + dtype * n_rows)

    def reader(f):
        return struct.unpack(f.read(struct.size))

    return reader


read_vdif_head: Callable = make_binary_reader(N_ROWS_VDIF_HEAD, UINT)
read_corr_head: Callable = make_binary_reader(N_ROWS_CORR_HEAD, UINT)
read_corr_data: Callable = make_binary_reader(N_ROWS_CORR_DATA, SHORT)


# struct parsers
def parse_vdif_head(vdif_head: list):
    # not implemented yet
    pass


def parse_corr_head(corr_head: list):
    # not implemented yet
    pass


def parse_corr_data(corr_data: list) -> np.ndarray:
    real = np.array(corr_data[0::2])
    imag = np.array(corr_data[1::2])
    return real + imag * 1j
