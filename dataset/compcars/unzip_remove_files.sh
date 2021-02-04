password="d89551fd190e38"
echo "comprimiendo en un zip"
zip -qq -F data.zip --out combined.zip 
rm data.z*
echo "descomprimiendo archivos"
unzip -qq -P $password  combined.zip 
rm combined.zip