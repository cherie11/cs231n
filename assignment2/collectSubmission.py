import os
os.system("rm -f assignment2.7z")
os.system("zip -r assignment2 . -xr!*cs231n/datasets* -xr!*ipynb_checkpoints* -xr!*README.md -xr!*collectSubmission* -xr!*requirements.txt -xr!*git* -xr!7z.exe")