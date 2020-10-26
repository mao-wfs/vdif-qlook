__all__ = ["get_spectrum"]


# standard library
from struct import Struct
from pathlib import Path


# dependent packages
import numpy as np


# constants
LITTLE_ENDIAN = "<"
UINT = "I"
SHORT = "h"
N_ROWS_VDIF_HEAD = 8
N_ROWS_CORR_HEAD = 64
N_ROWS_CORR_DATA = 512
N_UNITS_PER_SCAN = 64
N_BYTES_PER_UNIT = 1312
N_BYTES_PER_SCAN = 1312 * 64
TIME_PER_SCAN = 5e-3


# main features
def get_spectrum(path: Path, integ: float, chbin: int) -> np.ndarray:
    spectra = get_spectra(path, integ)
    spectrum = integrate_spectra(spectra, chbin)
    return spectrum


# sub features
def get_spectra(path: Path, integ: float) -> np.ndarray:
    n_scans = path.stat().st_size // N_BYTES_PER_SCAN
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


def integrate_spectra(spectra: np.ndarray, chbin: int) -> np.ndarray:
    spectrum = spectra.mean(0)
    return spectrum.reshape([len(spectrum) // chbin, chbin]).mean(1)


# readers
def make_binary_reader(n_rows, dtype):
    struct = Struct(LITTLE_ENDIAN + dtype * n_rows)

    def reader(f):
        return struct.unpack(f.read(struct.size))

    return reader


read_vdif_head = make_binary_reader(N_ROWS_VDIF_HEAD, UINT)
read_corr_head = make_binary_reader(N_ROWS_CORR_HEAD, UINT)
read_corr_data = make_binary_reader(N_ROWS_CORR_DATA, SHORT)


# parsers
def parse_vdif_head(vdif_head):
    # not implemented yet
    pass


def parse_corr_head(corr_head):
    # not implemented yet
    pass


def parse_corr_data(corr_data):
    real = np.array(corr_data[0::2])
    imag = np.array(corr_data[1::2])
    return real + imag * 1j
