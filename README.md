  -----------------------------------
  SWeeP: Sequence Window Projection
  -----------------------------------

This Python package implements the SWeeP (Sequence Window Projection)
algorithm for representing large biological sequence datasets in compact
vectors. SWeeP allows efficient processing and analysis of sequence data
by converting sequences into fixed-length feature vectors.

# Installation

To use SWeeP in Python, install the package with the following command:

    pip install sweep

# Usage

## Downloading the Default Projection Matrix

In the first use of fas2sweep with the default parameter, it will be
necessary to download the default projection matrix. It is not necessary
for use with custom projection matrix, as
demonstrated in the "Changing Projection Matrix" topic.

    from sweep import down_proj_mat
    down_proj_mat() # Downloads the default projection matrix file

## Handling Amino Acid Sequences

The default configurations of SWeeP are intended for vectorization of
amino acid sequences. The default output is a matrix already projected
with 600 columns. Here\'s an example of how to use SWeeP with amino acid
sequences:

    from sweep import fastaread, fas2sweep

    fasta = fastaread("fasta_file_path")
    vect = fas2sweep(fasta)

## Changing Projection Matrix

To change the projection matrix, a new orthonormal matrix can be
generated using the orthbase function. For example,
here is an example of how to change the projection size to 300:

    from sweep import fastaread, fas2sweep, orthbase

    ob = orthbase(160000, 300)
    fasta = fastaread("fasta_file_path")
    vect = fas2sweep(fasta, orth_mat=ob)

## Handling Nucleotide Sequences

For nucleotide sequences, there is no default projection matrix
available in this version. Therefore, to work with nucleotides
is possible to create a custom projection matrix using the
orthbase function. The matrix size can be calculated using
the calc_proj_mat_size function. Here is an example:

    from sweep import fastaread, fas2sweep, orthbase, calc_proj_mat_size

    mask = [4, 7, 4]
    matrix_size = calc_proj_mat_size(mask, 'NT')
    ob = orthbase(matrix_size, 600)
    fasta = fastaread("fasta_file_path")
    vect = fas2sweep(fasta, mask=mask, orth_mat=ob, fasta_type='NT')

## Available Functions

Here is a summary of the
functions available in the SWeeP package:

| Function                            | Description                                                                      | Input                                                          | Output                                          |
|-------------------------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------|
| ``fastaread``                       | Reads a FASTA file and returns a list of sequence records                        | ``fastaname`` (str): Path to the FASTA file                    | ``records`` (list): List of sequence records    |
| ``fas2sweep``                       | Converts a list of sequences into SWeeP vectors                                  | ``fasta`` (list): List of sequence records                     | ``vect`` (numpy.ndarray): SWeeP vectors         |
| ``orthbase``                        | Generates an orthonormal projection matrix of the specified size                 | ``lin`` (int): Number of rows                                  | ``mret`` (numpy.ndarray): Orthonormal matrix    |
| ``calc_proj_mat_size``              | Calculates the number of lines in the projection matrix for a given mask         | ``mask`` (list): Mask specifying dimensions                    | ``lines`` (int): Number of lines in the matrix  |
| ``md5``                             | Calculates the MD5 hash of a file                                                | ``fname`` (str): Path to the file                              | ``hash_md5`` (str): MD5 hash of the file        |
| ``down_proj_mat``                   | Downloads the default projection matrix file                                     | ``destination`` (str): Path to the destination file (optional) | None                                            |
| ``return_proj_mat_not_found_error`` | Raises an exception indicating that the default projection matrix is not found   | None                                                           | None                                            |
| ``check_default_proj_mat``          | Checks if the default projection matrix exists and matches the expected MD5 hash | ``file`` (str): Path to the projection matrix file             | None                                            |
| ``get_default_proj_mat``            | Retrieves the default projection matrix                                          | None                                                           | ``orth_mat`` (numpy.ndarray): Projection matrix |

## Article Reference

If you use the SWeeP algorithm or this Python package in your research work, please cite the
following article:

``` none
@article{De Pierri2020,
  title={SWeeP: representing large biological sequences datasets in compact vectors},
  author={De Pierri, Camilla Reginatto and Voyceik, Ricardo and Santos de Mattos, Letícia Graziela Costa and Kulik, Mariane Gonçalves and Camargo, Josué Oliveira and Repula de Oliveira, Aryel Marlus and de Lima Nichio, Bruno Thiago and Marchaukoski, Jeroniza Nunes and da Silva Filho, Antonio Camilo and Guizelini, Dieval and Ortega, J. Miguel and Pedrosa, Fabio O. and Raittz, Roberto Tadeu},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={91},
  year={2020},
  doi={10.1038/s41598-019-55627-4},
  url={https://doi.org/10.1038/s41598-019-55627-4},
  issn={2045-2322}
}
```

Please refer to the article for a detailed description of the SWeeP
algorithm and its applications.