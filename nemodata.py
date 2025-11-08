from dataclasses import dataclass
from collections import defaultdict
import warnings
import numpy as np

@dataclass
class NemoData:
    """
    Container for NEMO 3D output data.
    """
    data: np.ndarray
    meta: dict


def _read_header(file):
    """
    Read header from a binary NEMO 3D output file.

    Parameters
    ----------
    file : io.BufferedReader
        The opened binary file object.

    Returns
    -------
    list[str]
        The lines of the NEMO 3D output file header.

    Raises
    ------
    RuntimeError
        When the file format is not supported.

    """
    terminal = b'<\\HDR>'
    header = b''
    match = 0
    while True:
        if match == len(terminal):
            break
        char = file.read(1)
        if char == b'':
            raise RuntimeError('Unsupported file format.')
        if char == terminal[match:match + 1]:
            match += 1
        header += char
    return header.decode('ascii').splitlines()[:-1]


def _read_header_ascii(file):
    """
    Read header from an ASCII NEMO 3D output file.

    Parameters
    ----------
    file : io.TextIOWrapper
        The opened ASCII file object.

    Returns
    -------
    list[str]
        The lines of the NEMO 3D output file header.

    Raises
    ------
    RuntimeError
        When the file format is not supported, or a binary file is opened.

    """
    header_lines = []
    for line in file:
        if line == '<\\HDR>\n':
            break
        if line == '':
            raise RuntimeError('Unsupported file format.')
        if '<\\HDR>\n' in line:
            raise RuntimeError('A binary file is opened. Please use parse().')
        header_lines.append(line.rstrip('\n'))
    return header_lines


def _parse_header_lines(lines):
    """
    Convert header lines to metadata.

    Parameters
    ----------
    lines : list[str]
        The lines of the NEMO 3D output file header.

    Returns
    -------
    dict[str, list[tuple]]
        The dictionary representation of the metadata.

    """
    meta = {'record format': [], 'record dimension': [], 'user data': []}
    for line in lines:
        name, value = line.split('=', maxsplit=1)
        name = name.strip()
        if name == 'record format':
            types = value.strip(' {}').split(',')
            for t in types:
                t = t.strip().split(maxsplit=1)
                meta['record format'].append(tuple(reversed(t)))
        elif 'record dimension' in name:
            content = [v.strip() for v in value.split('#', maxsplit=1)]
            content[0] = int(content[0])
            meta['record dimension'].append(tuple(reversed(content)))
        elif 'user data' in name:
            content = [v.strip() for v in value.split('#', maxsplit=1)]
            if content[0].isdigit():
                content[0] = int(content[0])
            elif content[0].isdecimal():
                content[0] = float(content[0])
            meta['user data'].append(tuple(reversed(content)))
    return meta


def _recfmt_to_dtype(recfmt):
    """
    Create a NumPy dtype from a NEMO 3D output record format.

    Parameters
    ----------
    recfmt : list[tuple[str]]
        The 'record format' in metadata.

    Returns
    -------
    list[tuple[str]]
        An numpy.dtype compatible list.

    """
    datatype = []
    cmplx_name = ''
    cmplx_type = ''
    for name, fmt in recfmt:
        # complex treatment
        if name[-2:] == '.r':
            cmplx_name = name[:-2]
            cmplx_type = fmt
        elif name[-2:] == '.i' and name[:-2] == cmplx_name and fmt == cmplx_type:
            if fmt == 'real':
                datatype[-1] = (cmplx_name, 'c16')
            else:  # fmt == 'float' supposedly
                datatype[-1] = (cmplx_name, 'c8')
            continue
        else:
            # do not keep track of unpaired complex components
            cmplx_name = ''
            cmplx_type = ''

        if fmt == 'real':
            datatype.append((name, 'f8'))
        elif fmt == 'float':
            datatype.append((name, 'f4'))
        elif fmt == 'int':
            datatype.append((name, 'i4'))
    return datatype[0][1] if len(datatype) == 1 else np.dtype(datatype)


def parse(filename):
    """
    Parse a binary NEMO 3D output file.

    Parameters
    ----------
    filename : str
        The path to the binary NEMO 3D output file.

    Returns
    -------
    NemoData
        The container with parsed NEMO 3D output data.

    Raises
    ------
    RuntimeError
        When the file format is not supported.

    """
    with open(filename, 'rb') as file:
        # initial treatment
        if file.readline() != b'<HDR>\n':
            raise RuntimeError('Unsupported file format.')

        # parse header
        meta = _parse_header_lines(_read_header(file))

        # warnings
        if '_ascii' in filename:
            warnings.warn('The file is likely not a binary file. Consider using parse_ascii().',
                          RuntimeWarning)
        if '_dx+' in filename:
            warnings.warn('A dx file is opened. Special treatment may be required.',
                          RuntimeWarning)

        # parse data
        datatype = _recfmt_to_dtype(meta['record format'])
        data = np.fromfile(file, dtype=datatype)
        return NemoData(data, meta)


def parse_ascii(filename):
    """
    Parse an ASCII NEMO 3D output file.

    Parameters
    ----------
    filename : str
        The path to the ASCII NEMO 3D output file.

    Returns
    -------
    NemoData
        The record with parsed NEMO 3D output data.

    Raises
    ------
    RuntimeError
        When the file format is not supported.

    """
    with open(filename, 'r') as file:
        # detect header
        if file.readline() != '<HDR>\n':
            raise RuntimeError('Unsupported file format.')

        # parse header
        meta = _parse_header_lines(_read_header_ascii(file))

        # warnings
        if '_ascii' not in filename:
            warnings.warn('The file is likely a binary file. Consider using parse().',
                          RuntimeWarning)
        if '_dx+' in filename:
            warnings.warn('A dx file is opened. Special treatment may be required.',
                          RuntimeWarning)

        # parse data
        datatype = _recfmt_to_dtype(meta['record format'])
        datanames = [name for name, _ in meta['record format']]
        data = np.genfromtxt(file, names=datanames)  # create structured array
        data.dtype = datatype  # assign correct dtype
        return NemoData(data, meta)


def evec_reshape(record):
    """
    Reshape the NEMO 3D data parsed from a .nd_evec file.

    Parameters
    ----------
    record : NemoData
        The NEMO 3D record containing data from a .nd_evec file.

    Raises
    ------
    ValueError
        When the is not from a .nd_evec file.
    RuntimeError
        When the band model is invalid.

    Notes
    -----
    For band model with spin, +/- denote up-/down-spin respectively.

    """
    if (len(record.meta['record dimension']) != 2 or
        record.meta['record dimension'][0][1] *
        record.meta['record dimension'][1][1] != record.data.size):
        raise ValueError('The record is not from a .nd_evec file.')

    numtype = record.data.dtype
    if record.meta['user data'][0][1] == 'Bands_20_sp3d5ss_spin':
        record.data.dtype = [('s*-', numtype),
                             ('s-', numtype),
                             ('px-', numtype),
                             ('py-', numtype),
                             ('pz-', numtype),
                             ('dxy-', numtype),
                             ('dyz-', numtype),
                             ('dzx-', numtype),
                             ('dx2-y2-', numtype),
                             ('dz2-', numtype),
                             ('s*+', numtype),
                             ('s+', numtype),
                             ('px+', numtype),
                             ('py+', numtype),
                             ('pz+', numtype),
                             ('dxy+', numtype),
                             ('dyz+', numtype),
                             ('dzx+', numtype),
                             ('dx2-y2+', numtype),
                             ('dz2+', numtype)]
    elif record.meta['user data'][0][1] == 'Bands_10_sp3ss_spin':
        record.data.dtype = [('s*-', numtype),
                             ('s-', numtype),
                             ('px-', numtype),
                             ('py-', numtype),
                             ('pz-', numtype),
                             ('s+', numtype),
                             ('s*+', numtype),
                             ('px+', numtype),
                             ('py+', numtype),
                             ('pz+', numtype)]
    elif record.meta['user data'][0][1] == 'Bands_10_sp3d5ss_nospin':
        record.data.dtype = [('s*', numtype),
                             ('s', numtype),
                             ('px', numtype),
                             ('py', numtype),
                             ('pz', numtype),
                             ('dxy', numtype),
                             ('dyz', numtype),
                             ('dzx', numtype),
                             ('dx2-y2', numtype),
                             ('dz2', numtype)]
    elif record.meta['user data'][0][1] != 'Bands_1_s_nospin':
        raise RuntimeError('Unsupported band model.')


def project(coord, data, axis_1, axis_2=None):
    """
    Project the data onto the selected coordinate axes.

    Parameters
    ----------
    coord : numpy.ndarray
        The NumPy structured array of the coordinates.
    data : numpy.ndarray
        The NumPy array of the data.
    axis_1 : str
        The first axis to project onto.
    axis_2 : str, optional
        The second axis to project onto. If not provided, the projection is 1D.

    Returns
    -------
    numpy.ndarray
        A NumPy array of the unique coordinate(s) with
        the associated projected values.

    Raises
    ------
    ValueError
        When the two NumPy arrays has different dimension.
    TypeError
        When the value of the data array cannot be summed.
        This happens when a not numerical record is passed.

    """
    if coord.size != data.size:
        raise ValueError('The size of the coordinate and data arrays must be equal.')

    subcoord = coord[[axis_1, axis_2]] if axis_2 else coord[axis_1]
    unique_coord, indices = np.unique(subcoord, axis=0, return_inverse=True)
    projection = np.bincount(indices, weights=data)
    return (np.c_[unique_coord[axis_1], unique_coord[axis_2], projection]
            if axis_2 else np.c_[unique_coord, projection])
