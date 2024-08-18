@echo off


:loop
C:\ProgramData\anaconda3\python.exe C:\Users\Administrator\Desktop\GNN-AirComp-RIS\cvx_ris.py 
if %ERRORLEVEL% == 0 (
    echo over
    goto :end
) else (
    echo retry
    goto loop
)

:end
