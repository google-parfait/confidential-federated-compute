# Generate requirements.txt for the tff_worker

To ensure safe downloads in the case that an attacker compromises the PyPI
account of a library we depend on, we require hashes for all packages installed
by Pip.

This folder contains files necessary to create the requirements.txt file with
hashes, using the same version of Python as the Dockerfile in the parent
directory.

To actually regenerate requirements.txt, run the gen_requirements.sh shell
script in the parent directory, which will make use of the files in this folder.
