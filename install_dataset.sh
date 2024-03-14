curl "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip" -o proofwriter-dataset-V2020.12.3.zip
unzip proofwriter-dataset-V2020.12.3.zip
rm proofwriter-dataset-V2020.12.3.zip
mv proofwriter-dataset-V2020.12.3 proofwriter-dataset
echo "Dataset installed at proofwriter-dataset"
