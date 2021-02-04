password="d89551fd190e38"
echo "comprimiendo en un zip"
zip -F data.zip --out combined.zip
rm data.z*
echo "descomprimiendo archivos"
unzip -P $password  combined.zip
rm combined.zip