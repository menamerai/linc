Invoke-WebRequest -URI https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip -OutFile proofwriter-dataset-V2020.12.3.zip
Expand-Archive -Path proofwriter-dataset-V2020.12.3.zip -DestinationPath .
Remove-Item -Path proofwriter-dataset-V2020.12.3.zip
Rename-Item -Path .\proofwriter-dataset-V2020.12.3 -NewName .\proofwriter-dataset