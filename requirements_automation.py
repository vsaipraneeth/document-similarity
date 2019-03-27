import os
import re
import subprocess
scriptsDir=input("Enter Python Script Location:")#'C:/Python27/Scripts'
if os.path.exists(scriptsDir):
    os.chdir(scriptsDir)
    print("Modules Installation Begining:")
    print("#")
    print("#")
    print("#")
    print("#")      
    m_p=subprocess.Popen('pip install -U -r C:/Python27/MajorProject/requirements.txt',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,error=m_p.communicate()

    print(out)
    
    print("#")
    print("#")
    print("#")
    print("#")
    print("Modules Installation Done Successfully")
