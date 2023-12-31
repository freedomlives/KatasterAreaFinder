# KatasterAreaFinder

Slovakia now requires farmers to list what percentage of parcels they own is being used for fields where they grow crops, graze or cut hay. Since often these fields cover multiple parcels, and at the same time the entire parcel is not included in the field (e.g. after 70 years one end has grown into forest), it is a bit complicated to accurately say how many m^2 of each parcel is included in an agricultural field.

Kataster data in Slovakia is only publicly available as raster images from WMS servers. The data on declared agriculture fields (which farmers make thru use of GSAA web application) is available as a data file of vectors. My attempt to use QGIS utilities to calculate how much of each parcel is covered by a field were futile, so I wrote my own program, using maps manually extracted from QGIS.

To get the borders of parcels from registry E, I used this WMS server: https://kataster.skgeodesy.sk/eskn/services/NR/uo_wms_orto/MapServer/WmsServer Best would be to find one with just lines and not parcel numbers in the image, but the only one I found doesn't produce nice lines and for me OpenCV had trouble: https://kataster.skgeodesy.sk/eskn/rest/services/VRM/parcels_e_view/MapServer

The borders of fields (hranice užívania) are available from https://data.gov.sk/dataset/hranice-uzivania   I used the gml file for this year, and imported it to QGIS. *QGIS will ask to translate the coordinate system, the 4th option is correct for Slovakia* Copy each of your fields into separate layers, and color the polygons as pure red (program looks for red pixels).

With first just the field polygon visible, export the map to PNG, and then only with kataster visible. Ensure the entirety of each parcel you want to calculate percentage used for is included in the resulting image. Add for each field another polygon layer, and draw around everything I want to include, set the extent of that layer to drawn polygon, and then use that layer as reference when exporting from QGIS.

Example output, using the two input maps included in this repository:
![percents_calculated](https://github.com/freedomlives/KatasterAreaFinder/assets/16663872/73bf368f-5528-4c3e-8dd0-739bb28f9724)
