wget https://people.csail.mit.edu/mrub/ObjectDiscovery/ObjectDiscovery-data.zip
unzip ObjectDiscovery-data.zip
rm -r ObjectDiscovery-data.zip Results

cp -r Data/Airplane100 .
cp -r Data/Horse100 .
cp -r Data/Car100 .

rm -r Data

