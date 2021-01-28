password="d89551fd190e38"
zip -F data.zip --out combined.zip
rm data.z*
unzip -P $password  combined.zip
rm combined.zip