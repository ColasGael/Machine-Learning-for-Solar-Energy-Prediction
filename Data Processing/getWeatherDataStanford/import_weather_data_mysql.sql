/*Table structure for table `local_weather` */

DROP TABLE IF EXISTS `local_weather`;

CREATE TABLE `local_weather` (
  `id` int(6) unsigned NOT NULL AUTO_INCREMENT,
  `zip5` int(11) NOT NULL,
  `date` datetime NOT NULL,
  `TemperatureF` float(5,2) DEFAULT NULL,
  `DewpointF` float(5,2) DEFAULT NULL,
  `Pressure` float(5,2) DEFAULT NULL,
  `WindSpeed` float(5,2) DEFAULT NULL,
  `Humidity` float(5,2) DEFAULT NULL,
  `Clouds` varchar(10) DEFAULT NULL,
  `HourlyPrecip` float(5,2) DEFAULT NULL,
  `SolarRadiation` float(6,2) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `zip_date_idx` (`zip5`,`date`),
  KEY `zip_idx` (`zip5`),
  KEY `date_idx` (`date`)
) ENGINE=MyISAM DEFAULT CHARSET=latin1;

/* load weather data in the format produced by local_weather's dumpWeather.py utility */
LOAD DATA LOCAL INFILE 'your/path/to/weather_data_export.csv' 
	INTO TABLE local_weather 
	FIELDS TERMINATED BY ','  
	LINES TERMINATED BY '\n'
  IGNORE 1 LINES
	(@dateStr, TemperatureF, DewpointF, Pressure, WindSpeed, Humidity, HourlyPrecip, zip5)
	SET `date` = STR_TO_DATE(@dateStr, '%Y-%m-%d %H:%i:%s');


