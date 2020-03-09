Image processing library
========================

Brief description
-----------------

Utility which allows you to store and process field's images. Visualize images data via convenient command line and big amount of settings.

Requirements
------------

To use this utility you should have installed GDAL and rasterio >= 1.0.24

Utility features
================

Import
------

Note that to import file it follow naming convention to provide meta
information about file

File name pattern := TBD

-  Import picture
-  Import set of pictures
-  Import with caching statistical data

Export
------

-  Export images on a certain timeline by id
-  Export all data into folder
-  Export image statistics into excel file

Processing
----------

-  Calculate all statistical data and print it to screen
-  Caching calculations into database

Visualization
-------------

-  Histogram to visualize cloudiness of images on a timeline
-  Histogram to visualize image bits occurences of an image
-  Diagram to visualize multiple datasets statistical data

View database
-------------

-  Print out all database records onto screen
-  View head N records
-  Filter images by multiple ids
-  Filter images by date

